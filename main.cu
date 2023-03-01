// hello world cuda addition program

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <tiffio.h>

// nvcc -o kmeans kmeans.cu `pkg-config --cflags --libs opencv` -ltiff

using namespace std;
// N is the number of data points (which is the same as the number of pixels in the image)

// TPB is the number of threads per block
#define TPB 32

// K is the number of clusters
#define K 3

// MAX_ITER is the maximum number of iterations
#define MAX_ITER 10

// Euclidean distance between two colors, x1, y1, z1 are the coordinates of the first color, x2, y2, z2 are the coordinates of the second color (which is the centroid)
__device__ float distance(int x1, int y1, int z1, float x2, float y2, float z2)
{
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
}

// Assign each data point to the closest centroid
__global__ void kmeansAssign(int *d_datapoints, int *d_clust_assn, float *d_centroids, int *N)
{

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= *N)
    {
        return;
    }
    const int tid_x = tid * 3;

    float min_dist = INFINITY;
    int closest_centroid = -1;

    for (int i = 0; i < K; i++)
    {
        int i_x = i * 3;
        float dist = distance(d_datapoints[tid_x], d_datapoints[tid_x + 1], d_datapoints[tid_x + 2], d_centroids[i_x], d_centroids[i_x + 1], d_centroids[i_x + 2]);
        if (dist < min_dist)
        {
            min_dist = dist;
            closest_centroid = i;
        }
    }

    d_clust_assn[tid] = closest_centroid;
}

// Update the centroids. Each index 0 in the thread block will update a centroid
// `d_datapoints` is the array of data points (which is 3 elements long, and should be indexed by 3 * tid)
// `d_clust_assn` is the array of cluster assignments (which is 1 element long, and should be indexed by tid)
// `d_centroids` is the array of centroids (which is 3 elements long, and should be indexed by 3 * tid)
// `d_clust_sizes` is the array of cluster sizes (which is 1 element long, and should be indexed by tid)
__global__ void kmeansUpdate(int *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_clust_sizes, int *N)
{

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= *N)
    {
        return;
    }

    // tid_x is the index of the first element of the data point in the array (which is 3 elements long)
    const int tid_x = tid * 3;

    const int s_idx = threadIdx.x;

    __shared__ int s_datapoints[TPB * 3];
    s_datapoints[s_idx] = d_datapoints[tid_x];

    __shared__ int s_clust_assn[TPB];
    s_clust_assn[s_idx] = d_clust_assn[tid];

    __syncthreads();

    if (s_idx == 0)
    {
        float b_clust_datapoint_sums_r[K] = {0};
        float b_clust_datapoint_sums_g[K] = {0};
        float b_clust_datapoint_sums_b[K] = {0};

        int b_clust_sizes[K] = {0};

        for (int j = 0; j < blockDim.x; ++j)
        {
            int clust_id = s_clust_assn[j];
            // add the data point to the sum of the cluster (ints to floats)
            b_clust_datapoint_sums_r[clust_id] += s_datapoints[j * 3];
            printf("b_clust_datapoint_sums_r[%d] = %f\n", clust_id, b_clust_datapoint_sums_r[clust_id]);
            b_clust_datapoint_sums_g[clust_id] += s_datapoints[j * 3 + 1];
            printf("b_clust_datapoint_sums_g[%d] = %f\n", clust_id, b_clust_datapoint_sums_g[clust_id]);
            b_clust_datapoint_sums_b[clust_id] += s_datapoints[j * 3 + 2];
            printf("b_clust_datapoint_sums_b[%d] = %f\n", clust_id, b_clust_datapoint_sums_b[clust_id]);
            b_clust_sizes[clust_id]++;
        }

        // Now we add the sums to the global centroids and add the counts to the global counts.
        for (int z = 0; z < K; ++z)
        {
            int z_x = z * 3;
            atomicAdd(&d_centroids[z_x], b_clust_datapoint_sums_r[z_x]);
            atomicAdd(&d_centroids[z_x + 1], b_clust_datapoint_sums_g[z_x]);
            atomicAdd(&d_centroids[z_x + 2], b_clust_datapoint_sums_b[z_x]);
            atomicAdd(&d_clust_sizes[z], b_clust_sizes[z]);
        }
    }

    __syncthreads();

    // currently centroids are just sums, so divide by size to get actual centroids
    if (tid < K)
    {
        printf("d_clust_sizes[tid_x] = %d\n", d_clust_sizes[tid]);
        if (d_clust_sizes[tid] == 0)
        {
            return;
        }
        d_centroids[tid_x] = d_centroids[tid_x] / d_clust_sizes[tid];
        d_centroids[tid_x + 1] = d_centroids[tid_x + 1] / d_clust_sizes[tid];
        d_centroids[tid_x + 2] = d_centroids[tid_x + 2] / d_clust_sizes[tid];
    }
}

int main(void)
{

    string imageName("image.svs"); // start with a default

    TIFF *tif = TIFFOpen(imageName.c_str(), "r");

    // check the tif is open
    if (tif == NULL)
    {
        cerr << "Could not open tiff image" << endl;
        return -1;
    }

    unsigned int width, height = 0;

    // get the size of the tiff
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);


    int N = width * height; // get the total number of pixels
    int *d_N = 0;
    cudaMalloc(&d_N, sizeof(int));
    cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice);
    

    // allocate memory on the device for the data points
    int *d_datapoints = 0;
    cudaMalloc(&d_datapoints, N * sizeof(int) * 3);

    // allocate memory on the device for the cluster assignments
    int *d_clust_assn = 0;
    cudaMalloc(&d_clust_assn, N * sizeof(int));

    // allocate memory on the device for the cluster centroids
    float *d_centroids = 0;
    cudaMalloc(&d_centroids, K * sizeof(float) * 3);

    // allocate memory on the device for the cluster sizes
    int *d_clust_sizes = 0;
    cudaMalloc(&d_clust_sizes, K * sizeof(int));

    float *h_centroids = (float *)malloc(K * sizeof(float) * 3);

    int *h_datapoints = (int *)malloc(N * sizeof(int) * 3);
    int *h_clust_sizes = (int *)malloc(K * sizeof(int));
    int *h_clust_assn = (int *)malloc(N * sizeof(int));


    uint32 *raster;
    raster = (uint32 *)_TIFFmalloc(N * sizeof(uint32)); // allocate temp memory (must use the tiff library malloc)
    if (raster == NULL)                                       // check the raster's memory was allocaed
    {
        TIFFClose(tif);
        cerr << "Could not allocate memory for raster of TIFF image" << endl;
        return -1;
    }

    // Check the tif read to the raster correctly
    if (!TIFFReadRGBAImage(tif, width, height, raster, 0))
    {
        TIFFClose(tif);
        cerr << "Could not read raster of TIFF image" << endl;
        return -1;
    }

    // itterate through all the pixels of the tif
    int current_pixel = 0;
    for (uint x = 0; x < width; x++)
        for (uint y = 0; y < height; y++)
        {
            int d_idx = current_pixel * 3;

            // get the current pixel
            uint32 TiffPixel = raster[y * width + x];
            // get the pixel values
            int r = TIFFGetR(TiffPixel);
            int g = TIFFGetG(TiffPixel);
            int b = TIFFGetB(TiffPixel);
            if (r == 0 && g == 0 && b == 0)
            {
                continue;
            }
            // set the pixel values in the data points array
            h_datapoints[d_idx + 0] = r;
            h_datapoints[d_idx + 1] = g;
            h_datapoints[d_idx + 2] = b;
            printf("Data point %d: R: %d, G: %d, B: %d \n", current_pixel, r, g, b);

            current_pixel++;
        }

    _TIFFfree(raster); // release temp memory

    TIFFClose(tif); // close the tif file
    



    srand(time(0));

    // initialize centroids
    for (int c = 0; c < K; ++c)
    {
        int c_idx = c * 3;
        // select three random numbers between 0 and 255
        int x_sel = rand() % 256;
        int y_sel = rand() % 256;
        int z_sel = rand() % 256;
        h_centroids[c_idx] = (float)x_sel;
        h_centroids[c_idx + 1] = (float)y_sel;
        h_centroids[c_idx + 2] = (float)z_sel;
        h_clust_sizes[c] = 0;
    }

    cudaMemcpy(d_centroids, h_centroids, K * sizeof(float) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_datapoints, h_datapoints, N * sizeof(int) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_clust_sizes, h_clust_sizes, K * sizeof(int), cudaMemcpyHostToDevice);

    int cur_iter = 1;

    while (cur_iter < MAX_ITER)
    {
        // call cluster assignment kernel
        kmeansAssign<<<(N + TPB - 1) / TPB, TPB>>>(d_datapoints, d_clust_assn, d_centroids, d_N);

        // copy new centroids back to host
        cudaMemcpy(h_centroids, d_centroids, K * sizeof(float) * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_clust_assn, d_clust_assn, N * sizeof(int), cudaMemcpyDeviceToHost);

        /*for (int i = 0; i < K; ++i)
        {
            printf("Iteration %d: centroid %d: R: %f, G: %f, B: %f \n", cur_iter, i, h_centroids[i * 3], h_centroids[(i * 3) + 1], h_centroids[(i * 3) + 2]);
        }*/

        printf("assignment: \n");
        // iterate over d_datapoints and d_clust_assn and print out the datapoints and their cluster assignments
        for (int i = 0; i < N; ++i)
        {
            int idx = i * 3;
            //printf("PIteration %d: Data point %d: R: %d, G: %d, B: %d, Cluster: %d \n", cur_iter, i, h_datapoints[idx], h_datapoints[idx + 1], h_datapoints[idx + 2], h_clust_assn[i]);
        }

        printf("\n");
        // reset centroids and cluster sizes (will be updated in the next kernel)
        cudaMemset(d_centroids, 0.0, K * sizeof(float) * 3);
        cudaMemset(d_clust_sizes, 0, K * sizeof(int));

        // call centroid update kernel
        kmeansUpdate<<<(N + TPB - 1) / TPB, TPB>>>(d_datapoints, d_clust_assn, d_centroids, d_clust_sizes, d_N);

        cur_iter += 1;
    }

    for (int i = 0; i < K; ++i)
    {
        printf("Iteration %d: centroid %d: R: %f, G: %f, B: %f \n", cur_iter, i, h_centroids[i * 3], h_centroids[(i * 3) + 1], h_centroids[(i * 3) + 2]);
    }

    cudaFree(d_datapoints);
    cudaFree(d_clust_assn);
    cudaFree(d_centroids);
    cudaFree(d_clust_sizes);
    cudaFree(d_N);


    // ouput three png files for each cluster
    for (int i = 0; i < K; ++i)
    {
        char * filename = new char[20];
        sprintf(filename, "cluster%d.tiff", i);
        // create a new tif file
        TIFF *out = TIFFOpen(filename, "w");
        if (out == NULL)
        {
            cerr << "Could not open " << filename << " for writing" << endl;
            return -1;
        }

        // set the image width, height, and depth
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 3);
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, height);

        // allocate memory for the raster
        uint32 *out_raster;
        out_raster = (uint32 *)_TIFFmalloc(TIFFStripSize(out));
        if (out_raster == NULL)
        {
            TIFFClose(out);
            cerr << "Could not allocate memory for raster of TIFF image" << endl;
            return -1;
        }
        tstrip_t strip;

        // iterate over the strip and write the data
        for(strip = 0; strip < TIFFNumberOfStrips(out); strip++)
        {
            // iterate over the pixels in the strip
            for (int i = 0; i < width * height; ++i)
            {
                int idx = i * 3;
                if (h_clust_assn[i] == i)
                {
                    out_raster[i] = (h_datapoints[idx] << 16) | (h_datapoints[idx + 1] << 8) | h_datapoints[idx + 2];
                }
                else
                {
                    out_raster[i] = 0;
                }
            }
            if (TIFFWriteEncodedStrip(out, strip, out_raster, width) < 0)
            {
                cerr << "Could not write strip " << strip << " of " << filename << endl;
                return -1;
            }
        }

        // close the tif file
        TIFFClose(out);
        _TIFFfree(out_raster);
    }
    
    

    free(h_centroids);
    free(h_datapoints);
    free(h_clust_sizes);
    free(h_clust_assn);
    



    return 0;
}
