
__device__ int2 getNeigVec(const int &i)
{   //liefert für einen Index zwischen 0 und 3 die Position der nächsten Nachbarn --> Uhrzeigersinn
    switch(i) {
    case 0 : return make_int2(-1,0);
    case 1 : return make_int2(0,1);
    case 2 :  return make_int2(1,0);
    case 3 : return make_int2(0,-1);
    }
    return make_int2(1,0);
}

__device__ bool isinArray(int* array, int size, int element) {
    for (int i = 0; i < size; i++) {
        if (array[i] == element) {
            return true;
        }
    }
    return false;
}



//main Kernel
/*
Dies ist der Kern der Simulation. Er wird für jeden Zeitschritt aufgerufen und berechnet die Bewegung der Spins durch die 
Landau-Lifshitz-Gilbert Gleichung, die in der Funktion SimStep implementiert ist. Da dies eine differentialgleichung der 1. Ordnung ist,
wird sie hinreichend mittels des Euler-Verfahren gelöst. Der gesamte Code ist in CUDA C geschrieben und wird damit auf der GPU ausgeführt.
*/
texture<int, 3, cudaReadModeElementType> NNs; //Textur für v_s

__global__ void calc_step(float *Phi, bool *mask)
{ //Programm läuft über verschiedene Threads die parallel auf GPU laufen
    // index runs up from bottom left first, then starts at next column and runs up again
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x; // flattened 2D block and grid
    int y     = index % field_size_y;
    int x     = (index - y) / field_size_y;
    // int index_mask = x + y * field_size_x;

    // erstelle einen Fade in Phi von index 0 bis index 20000
    if(index < field_size_x * field_size_y)
    {
        if(mask[index])
        {
            if(isinArray(input_left, input_size, index))
            {
                
                // if(x == 2)
                // {
                //     int current_No_NNs = 0;
                //     for (int i = 0; i < No_NNs; i++)
                //     {
                //         int NN = tex3D(NNs, i, y, x);
                //         int is_unique = 1; // flag to check if NN is unique
                //         for (int j = 0; j < i; j++)
                //         {
                //             if (NN == tex3D(NNs, j, y, x))
                //             {
                //                 is_unique = 0; // if a duplicate is found, set flag to 0
                //                 break;
                //             }
                //         }
                //         if (is_unique) // if no duplicate is found after the inner loop, increment current_No_NNs
                //         {
                //             current_No_NNs++;
                //         }
                //     }
                //     if (y == 60)
                //     {
                //         printf("NN_1: %d\n", tex3D(NNs, 0, y, x));
                //         printf("NN_2: %d\n", tex3D(NNs, 1, y, x));
                //         printf("NN_3: %d\n", tex3D(NNs, 2, y, x));
                //         printf("NN_4: %d\n", tex3D(NNs, 3, y, x));
                //         printf("NN_5: %d\n", tex3D(NNs, 4, y, x));
                //         printf("NN_6: %d\n", tex3D(NNs, 5, y, x));
                //     }
                //     printf("(x, y), No_NNS, current No_NNs: (%d, %d) %d, %d\n", x, y, No_NNs, current_No_NNs);
                // }
                Phi[index] = j_default + Phi[index + field_size_y];
                return;
            }
            else if(isinArray(output_right, output_size, index))
            {
                Phi[index] = Phi[index - field_size_y] - j_default;
                return;
            }
            else    // NNs and No_NNs via for i in range(No_NNs) tex3D(NNs, i, y, x)
            {
                // // // Schritt 1: Berechne die Indizes der Nachbarn und die Anzahl der Nachbarn
                // int NN_indices[8]; //Array mit den Indizes der Nachbarn --> x,y,x,y,x,y,x,y
                // int sum_NN = 0; //Anzahl der Nachbarn

                // for(int i = 0; i < 4; i++)
                // {
                //     int NN_index = index + getNeigVec(i).y + getNeigVec(i).x * field_size_x ;
                //     if(mask[NN_index])
                //     {
                //         sum_NN++;
                //         NN_indices[i * 2] = getNeigVec(i).y;  // to add up later all the indices for replace_index
                //         NN_indices[i * 2 + 1] = getNeigVec(i).x;  // to add up later all the indices for replace_index
                //     }
                //     else
                //     {
                //         NN_indices[i * 2] = 0;   // to add up later all the indices for replace_index
                //         NN_indices[i * 2 + 1]  = 0;   // to add up later all the indices for replace_index
                //     }
                // }
                
                // Schritt 2: Teile in die Fälle auf ob index in der Mitte oder am Rand/in der Ecke liegt
                // falls index in der Mitte des Feldes liegt
                // // TEST WITH ALL POINTS 

                // if(x == 148)    // 6 NN
                // {
                //     if(y == 119)
                //     {
                //         printf("(x, y), No_NNS, current_No_NNs: (%d, %d) %d, %d\n", x, y, No_NNs, current_No_NNs);
                //     }
                // }
                // if(x == 150)    // 4 NN
                // {
                //     if(y == 119)
                //     {
                //         printf("(x, y), No_NNS, current_No_NNs: (%d, %d) %d, %d\n", x, y, No_NNs, current_No_NNs);
                //     }
                // }

                // if(x == 149)    // 5 NN
                // {
                //     if(y == 119)
                //     {
                //         printf("(x, y), No_NNS, current_No_NNs: (%d, %d) %d, %d\n", x, y, No_NNs, current_No_NNs);
                //     }
                // }

                // if(x == 148)    // 3 NN
                // {
                //     if(y == 118)
                //     {
                //         printf("(x, y), No_NNS, current_No_NNs: (%d, %d) %d, %d\n", x, y, No_NNs, current_No_NNs);
                //     }
                // }
                
                // if(x == 209)    // 2 NN
                // {
                //     if(y == 83)
                //     {
                //         printf("(x, y), No_NNS, current_No_NNs: (%d, %d) %d, %d\n", x, y, No_NNs, current_No_NNs);
                        // printf("NN_1: %d\n", tex3D(NNs, 0, y, x));
                        // printf("NN_2: %d\n", tex3D(NNs, 1, y, x));
                        // printf("NN_3: %d\n", tex3D(NNs, 2, y, x));
                        // printf("NN_4: %d\n", tex3D(NNs, 3, y, x));
                        // printf("NN_5: %d\n", tex3D(NNs, 4, y, x));
                        // printf("NN_6: %d\n", tex3D(NNs, 5, y, x));
                //     }
                // }
                float Phi_NN_sum = (0.0f); //Phi_NN_sum ist die Summe der Eigenvektoren der existierenden NN

                for (int i = 0; i < No_NNs; i++)
                {
                    Phi_NN_sum = Phi_NN_sum + Phi[tex3D(NNs, i, y, x)];
                }
                // Phi as the average of the NNs
                Phi[index] = ((1.0f) - omega_rel) * Phi[index] + \
                    omega_rel * Phi_NN_sum / (float)No_NNs;

                return;
            }
        }
    }
}