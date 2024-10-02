

// Setup Kernel

/*
Erster Teil des Kernels, welcher die Funktionen und Variablen enthaelt, welche fuer die
Simulation der Skyrmionen benoetigt werden.
*/

// #define block_size 64

__device__ float3 operator+(const float3 &a, const float3 &b)
{ // Funktion, welche zwei Vektoren addiert
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3 &a, const float3 &b)
{ // Funktion, welche zwei Vektoren addiert
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator/(const float3 &a, const int &b)
{ // Funktion, welche einen Vektor durch einen Integer teilt
  return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ float3 operator*(const float &a, const float3 &b)
{ // multiplies scalar with vector
  return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ float3 operator*=(const int &a, const float3 &b)
{ // multiplies scalar with vector
  return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ float dot(const float3 &a, const float3 &b)
{ // Skalarprodukt zweier Vektoren
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize(const float3 &a)
{ // normalisiert Vektor
  return rsqrt(dot(a, a)) * a;
}

__device__ float length(const float3 &a)
{ // normalisiert Vektor
  return sqrtf(dot(a, a));
}

__device__ float3 cross(const float3 &a, const float3 &b)
{ // Kreuzprodukt zweier Vektoren
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ float ComputeTriangleCharge(float3 r_1, float3 r_2, float3 r_3, float3 r_4) {
    float N = dot(r_1, cross(r_2, r_3));
    float D = 1.0f + dot(r_1, r_2) + dot(r_2, r_3) + dot(r_1, r_3);
    return 2.0f * atan2f(N, D);
    // return 1.0f;
}


// __device__ float total_sum = 0.0f;



// main Kernel
/*
Dies ist der Kern der Simulation. Er wird fuer jeden Zeitschritt aufgerufen und berechnet die Bewegung der Spins durch die
Landau-Lifshitz-Gilbert Gleichung, die in der Funktion SimStep implementiert ist. Da dies eine differentialgleichung der 1. Ordnung ist,
wird sie hinreichend mittels des Euler-Verfahren geloest. Der gesamte Code ist in CUDA C geschrieben und wird damit auf der GPU ausgefuehrt.
*/


texture<float, 3, cudaReadModeElementType> v_s; // Ensure this is initialized and bound correctly in host code


// __device__ float3 calculateBeff(const float3* spin, int index, float zComponent, const float3* kArray) {
//     float3 B_eff_aniso = 2 * K * zComponent * make_float3(0, 0, 1);
//     float3 B_eff_ext = B_ext * make_float3(0, 0, 1);
//     float3 B_eff_DM = make_float3(0, 0, 0);
//     float3 B_eff_exch = make_float3(0, 0, 0);

//     for (int i = 0; i < 4; i++) {
//         int neig_idx = index + NN_vec[i * 3 + 1] + NN_vec[i * 3] * size_y;

//         B_eff_exch = B_eff_exch + A * 0.5f * (spin[neig_idx] + 0.5f * kArray[neig_idx]);
//         B_eff_DM = B_eff_DM + cross(spin[neig_idx] + 0.5f * kArray[neig_idx], DM * 0.5f * make_float3(DM_vec[i * 3], DM_vec[i * 3 + 1], DM_vec[i * 3 + 2]));
//     }

//     return B_eff_aniso + B_eff_ext + B_eff_DM + B_eff_exch;
// }

// __device__ float3 calculateT(const float3* spin, int index, const float3* kArray, float coefficient) {
//     float3 T = make_float3(0, 0, 0); 
    
//     float3 tempSpin = spin[index] + coefficient * kArray[index];

//     int y     = index % size_y;
//     int x     = (index - y) / size_y;

//     for (int i = 0; i < 4; i++) {
//         int neig_idx = index + NN_vec[i * 3 + 1] + NN_vec[i * 3] * size_y;

//         float v_s_component = (i == 0 || i == 1) ? tex3D(v_s, 0, y, x) 
//                                                 : tex3D(v_s, 1, y, x);

//         T = T + v_s_component * (NN_vec[i * 3] + NN_vec[i * 3 + 1]) * (tempSpin + coefficient * kArray[neig_idx]) / 2.0f;
//     }

//     return T;
// }

// __device__ float3 calculateSlope(const float3* spin, int index, const float3& B_eff, const float3* kArray, float coefficient, float3 T, int* v_s_active) {
//     float3 tempSpin = spin[index] + coefficient * kArray[index];
//     return normalize(tempSpin + (-1.0f) / (1.0f + alpha * alpha) * dt * (gamma_el * cross(tempSpin, B_eff + cross(alpha * tempSpin, B_eff)) 
//             + (*v_s_active) * (-1) * (alpha - beta) * cross(tempSpin, T) + (*v_s_active) * (-1) * (alpha * beta + 1) * T)) + (-1) * tempSpin;
// }

// __global__ void SimStep(float3 *spin, bool *mask, float3 *k1, float3 *k2, float3 *k3, float3 *k4, int *v_s_active) {
//     int index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

//     if (index < size_x * size_y && mask[index]) {
//         k1[index] = calculateSlope(spin, index, calculateBeff(spin, index, spin[index].z, k1), k1, 0.0f, calculateT(spin, index, k1, 0.0f), v_s_active);
//         __syncthreads();
        
//         k2[index] = calculateSlope(spin, index, calculateBeff(spin, index, spin[index].z + 0.5f * k1[index].z, k1), k1, 0.5f, calculateT(spin, index, k1, 0.5f), v_s_active);
//         __syncthreads();
        
//         k3[index] = calculateSlope(spin, index, calculateBeff(spin, index, spin[index].z + 0.5f * k2[index].z, k2), k2, 0.5f, calculateT(spin, index, k2, 0.5f), v_s_active);
//         __syncthreads();
        
//         k4[index] = calculateSlope(spin, index, calculateBeff(spin, index, spin[index].z + k3[index].z, k3), k3, 1.0f, calculateT(spin, index, k3, 1.0f), v_s_active);
//         __syncthreads();

//         spin[index] = spin[index] + (k1[index] + 2.0f * k2[index] + 2.0f * k3[index] + k4[index]) / 6.0f;
//     }
// }

__global__ void SimStep_k1(float3 *spin, bool *mask, float3 *k1, int *v_s_active)                                                            // mit , float3 *B_tot am Ende wenn log von B_eff gewuenscht
{                                                                                                                           // Programm laeuft ueber verschiedene Threads die parallel auf GPU laufen
  
  int index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;  // flattened 2D block and grid
  int y     = index % size_y;
  int x     = (index - y) / size_y;

  if (index < size_x * size_y && mask[index]) // Pruefung mit Maske ob Spin simuliert werden muss
  {
    // ----------------------------------------------------------------- Calculation of k1 for RK4 -----------------------------------------------------------------
    // --> slope at the beginning of the interval
    // Definition aller B_eff --> DM und Exch. durch Schleife ueber alle NN
    float3 B_eff_aniso = 2 * K * spin[index].z * make_float3(0, 0, 1);
    float3 B_eff_ext   = B_ext * make_float3(0, 0, 1);                  // externes Magnetfeld
    float3 B_eff_DM    = make_float3(0, 0, 0);                          // DM-Vektor
    float3 B_eff_exch  = make_float3(0, 0, 0);                          // Austausch-WW

    // Spin Torque
    float3 T = make_float3(0, 0, 0);

    // Auswahl des richtigen NN-Vektors
    // Die differenzierung ist nur nötig falls es ein hexagonales Gitter ist 
    int *NN_pos_current;
    if (y % 2 == 0)
    {
      NN_pos_current = NN_pos_even_row;
    }
    else
    {
      NN_pos_current = NN_pos_odd_row;
    }

    for (int i = 0; i < No_NNs; i++) // alle naechsten Nachbarn
    {
      // Index des NN
      int neig_idx = index + NN_pos_current[i * 3 + 1] + NN_pos_current[i * 3] * size_y;

      // Austausch-WW:  J * Spin__NN
      B_eff_exch = B_eff_exch + A * (spin[neig_idx]);

      // DM-WW:  Spin_NN x DM_Vec
      B_eff_DM = B_eff_DM + cross(spin[neig_idx], DM * make_float3(DM_vec[i * 3], DM_vec[i * 3 + 1], DM_vec[i * 3 + 2]));

      // // Spin-Torque: (v_s * nabla) * Spin  --> dSpin/dx = (Spin(x+1) - Spin(x-1)) / 2
      // T = T + (i == 0 || i == 2 ? tex3D(v_s, 0, y, x) : tex3D(v_s, 1, y, x)) * (NN_vec[i * 3] + NN_vec[i * 3 + 1]) * spin[neig_idx] / 2.0f; // ternaerer Operator (Bedingung? Wert1 : Wert2) um die richtige Geschwindigkeitskomponente zu waehlen

      // Spin-Torque: (v_s * nabla) * Spin  --> dSpin/dx = (Spin(x+1) - Spin(x-1)) / 2
      T = T + ((tex3D(v_s, 0, y, x) * NN_vec[i * 3]) + (tex3D(v_s, 1, y, x) * NN_vec[i * 3 + 1])) * spin[neig_idx] / 2.0f; // ternaerer Operator (Bedingung? Wert1 : Wert2) um die richtige Geschwindigkeitskomponente zu waehlen
    }

    // B_eff log
    float3 B_eff = B_eff_aniso + B_eff_ext + B_eff_DM + B_eff_exch;

    // slope using euler for RK4
    k1[index] = normalize(spin[index] + (-1.0) / (1.0 + alpha * alpha) * dt *
                                              (gamma_el * cross(spin[index], B_eff + cross(alpha * spin[index], B_eff)) +
                                               (*v_s_active) * (-1) * (alpha - beta) * cross(spin[index], T) +
                                               (*v_s_active) * (-1) * (alpha * beta + 1) * T))
                + (-1.0) * spin[index];
  }
}

__global__ void SimStep_k2(float3 *spin, bool *mask, float3 *k1, float3 *k2, int *v_s_active)                                                            // mit , float3 *B_tot am Ende wenn log von B_eff gewuenscht
{                                                                                                                           // Programm laeuft ueber verschiedene Threads die parallel auf GPU laufen
  
  int index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;  // flattened 2D block and grid
  int y     = index % size_y;
  int x     = (index - y) / size_y;

  if (index < size_x * size_y && mask[index]) // Pruefung mit Maske ob Spin simuliert werden muss
  {
    
    // ----------------------------------------------------------------- Calculation of k2 for RK4 -----------------------------------------------------------------
    // slope in the middle of the interval

    // Definition aller B_eff --> DM und Exch. durch Schleife ueber alle NN
    float3 B_eff_aniso_mid = 2 * K * (spin[index].z + 0.5 * k1[index].z) * make_float3(0, 0, 1);
    float3 B_eff_ext_mid   = B_ext * make_float3(0, 0, 1);                  // externes Magnetfeld
    float3 B_eff_DM_mid    = make_float3(0, 0, 0);                          // DM-Vektor
    float3 B_eff_exch_mid  = make_float3(0, 0, 0);                          // Austausch-WW

    // Spin Torque
    float3 T_mid = make_float3(0, 0, 0);


    // Auswahl des richtigen NN-Vektors
    // Die differenzierung ist nur nötig falls es ein hexagonales Gitter ist 
    int *NN_pos_current;
    if (y % 2 == 0)
    {
      NN_pos_current = NN_pos_even_row;
    }
    else
    {
      NN_pos_current = NN_pos_odd_row;
    }

    for (int i = 0; i < No_NNs; i++) // alle naechsten Nachbarn
    {
      // Index des NN
      int neig_idx = index + NN_pos_current[i * 3 + 1] + NN_pos_current[i * 3] * size_y;

      // Austausch-WW:  J * Spin__NN
      B_eff_exch_mid = B_eff_exch_mid + A * (spin[neig_idx] + 0.5 * k1[neig_idx]);

      // DM-WW:  Spin_NN x DM_Vec
      B_eff_DM_mid = B_eff_DM_mid + cross(spin[neig_idx] + 0.5 * k1[neig_idx], DM * make_float3(DM_vec[i * 3], DM_vec[i * 3 + 1], DM_vec[i * 3 + 2]));

      // Spin-Torque: (v_s * nabla) * Spin  --> dSpin/dx = (Spin(x+1) - Spin(x-1)) / 2
      T_mid = T_mid + ((tex3D(v_s, 0, y, x) * NN_vec[i * 3]) + (tex3D(v_s, 1, y, x) * NN_vec[i * 3 + 1])) * (spin[neig_idx] + 0.5 * k1[neig_idx]) / 2.0f;
    }

    // B_eff log
    float3 B_eff_mid = B_eff_aniso_mid + B_eff_ext_mid + B_eff_DM_mid + B_eff_exch_mid;

    // slope using euler for RK4
    k2[index] = normalize(spin[index] + 0.5 * k1[index] + (-1.0) / (1.0 + alpha * alpha) * dt *
                                              (gamma_el * cross(spin[index] + 0.5 * k1[index], B_eff_mid + cross(alpha * (spin[index] + 0.5 * k1[index]), B_eff_mid)) +
                                               (*v_s_active) * (-1) * (alpha - beta) * cross(spin[index] + 0.5 * k1[index], T_mid) +
                                               (*v_s_active) * (-1) * (alpha * beta + 1) * T_mid))
                + (-1.0) * (spin[index] + 0.5 * k1[index]);
  }
}

__global__ void SimStep_k3(float3 *spin, bool *mask, float3 *k2, float3 *k3, int *v_s_active)                                                            // mit , float3 *B_tot am Ende wenn log von B_eff gewuenscht
{                                                                                                                           // Programm laeuft ueber verschiedene Threads die parallel auf GPU laufen
  int index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;  // flattened 2D block and grid
  int y     = index % size_y;
  int x     = (index - y) / size_y;

  if (index < size_x * size_y && mask[index]) // Pruefung mit Maske ob Spin simuliert werden muss
  {
    // ----------------------------------------------------------------- Calculation of k3 for RK4 -----------------------------------------------------------------
    // slope in the middle of the interval but using k2
    

    // Definition aller B_eff --> DM und Exch. durch Schleife ueber alle NN
    float3 B_eff_aniso_mid_2 = 2 * K * (spin[index].z + 0.5 * k2[index].z) * make_float3(0, 0, 1);
    float3 B_eff_ext_mid_2   = B_ext * make_float3(0, 0, 1);                  // externes Magnetfeld
    float3 B_eff_DM_mid_2    = make_float3(0, 0, 0);                          // DM-Vektor
    float3 B_eff_exch_mid_2  = make_float3(0, 0, 0);                          // Austausch-WW

    // Spin Torque
    float3 T_mid_2 = make_float3(0, 0, 0);


    // Auswahl des richtigen NN-Vektors
    // Die differenzierung ist nur nötig falls es ein hexagonales Gitter ist 
    int *NN_pos_current;
    if (y % 2 == 0)
    {
      NN_pos_current = NN_pos_even_row;
    }
    else
    {
      NN_pos_current = NN_pos_odd_row;
    }

    for (int i = 0; i < No_NNs; i++) // alle naechsten Nachbarn
    {
      // Index des NN
      int neig_idx = index + NN_pos_current[i * 3 + 1] + NN_pos_current[i * 3] * size_y;

      // Austausch-WW:  J * Spin__NN
      B_eff_exch_mid_2 = B_eff_exch_mid_2 + A * (spin[neig_idx] + 0.5 * k2[neig_idx]);

      // DM-WW:  Spin_NN x DM_Vec
      B_eff_DM_mid_2 = B_eff_DM_mid_2 + cross(spin[neig_idx] + 0.5 * k2[neig_idx], DM * make_float3(DM_vec[i * 3], DM_vec[i * 3 + 1], DM_vec[i * 3 + 2]));

      // Spin-Torque: (v_s * nabla) * Spin  --> dSpin/dx = (Spin(x+1) - Spin(x-1)) / 2
      T_mid_2 = T_mid_2 + ((tex3D(v_s, 0, y, x) * NN_vec[i * 3]) + (tex3D(v_s, 1, y, x) * NN_vec[i * 3 + 1])) * (spin[neig_idx] + 0.5 * k2[neig_idx]) / 2.0f; // ternaerer Operator (Bedingung? Wert1 : Wert2) um die richtige Geschwindigkeitskomponente zu waehlen
    }

    // B_eff log
    float3 B_eff_mid_2 = B_eff_aniso_mid_2 + B_eff_ext_mid_2 + B_eff_DM_mid_2 + B_eff_exch_mid_2;

    // slope using euler for RK4
    k3[index] = normalize(spin[index] + 0.5 * k2[index] + (-1.0) / (1.0 + alpha * alpha) * dt *
                                              (gamma_el * cross(spin[index] + 0.5 * k2[index], B_eff_mid_2 + cross(alpha * (spin[index] + 0.5 * k2[index]), B_eff_mid_2)) +
                                               (*v_s_active) * (-1) * (alpha - beta) * cross(spin[index] + 0.5 * k2[index], T_mid_2) +
                                               (*v_s_active) * (-1) * (alpha * beta + 1) * T_mid_2))
                + (-1.0) * (spin[index] + 0.5 * k2[index]);
  }
}

__global__ void SimStep_k4_nextspin(float3 *spin, bool *mask, float3 *k1, float3 *k2, float3 *k3, int *v_s_active)                                                            // mit , float3 *B_tot am Ende wenn log von B_eff gewuenscht
{                                                                                                                           // Programm laeuft ueber verschiedene Threads die parallel auf GPU laufen
  
  int index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;  // flattened 2D block and grid
  int y     = index % size_y;
  int x     = (index - y) / size_y;

  if (index < size_x * size_y && mask[index]) // Pruefung mit Maske ob Spin simuliert werden muss
  {
    // ----------------------------------------------------------------- Calculation of k4 for RK4 -----------------------------------------------------------------
    // slope at the end of the interval using k3

    // Definition aller B_eff --> DM und Exch. durch Schleife ueber alle NN
    float3 B_eff_aniso_end = 2 * K * (spin[index].z + k3[index].z) * make_float3(0, 0, 1);
    float3 B_eff_ext_end   = B_ext * make_float3(0, 0, 1);                  // externes Magnetfeld
    float3 B_eff_DM_end    = make_float3(0, 0, 0);                          // DM-Vektor
    float3 B_eff_exch_end  = make_float3(0, 0, 0);                          // Austausch-WW

    // Spin Torque
    float3 T_end = make_float3(0, 0, 0);


    // Auswahl des richtigen NN-Vektors
    // Die differenzierung ist nur nötig falls es ein hexagonales Gitter ist 
    int *NN_pos_current;
    if (y % 2 == 0)
    {
      NN_pos_current = NN_pos_even_row;
    }
    else
    {
      NN_pos_current = NN_pos_odd_row;
    }

    for (int i = 0; i < No_NNs; i++) // alle naechsten Nachbarn
    {
      // Index des NN
      int neig_idx = index + NN_pos_current[i * 3 + 1] + NN_pos_current[i * 3] * size_y;

      // Austausch-WW:  J * Spin__NN
      B_eff_exch_end = B_eff_exch_end + A * (spin[neig_idx] + k3[neig_idx]);

      // DM-WW:  Spin_NN x DM_Vec
      B_eff_DM_end = B_eff_DM_end + cross(spin[neig_idx] + k3[neig_idx], DM * make_float3(DM_vec[i * 3], DM_vec[i * 3 + 1], DM_vec[i * 3 + 2]));

      // Spin-Torque: (v_s * nabla) * Spin  --> dSpin/dx = (Spin(x+1) - Spin(x-1)) / 2
      T_end = T_end + ((tex3D(v_s, 0, y, x) * NN_vec[i * 3]) + (tex3D(v_s, 1, y, x) * NN_vec[i * 3 + 1])) * (spin[neig_idx] + k3[neig_idx]) / 2.0f; // ternaerer Operator (Bedingung? Wert1 : Wert2) um die richtige Geschwindigkeitskomponente zu waehlen
    }

    // B_eff log
    float3 B_eff_end = B_eff_aniso_end + B_eff_ext_end + B_eff_DM_end + B_eff_exch_end;

    // slope using euler for RK4  --> no k4 big array needed because of no extra calculation from k4 done
    float3 k4 = normalize(spin[index] + k3[index] + (-1.0) / (1.0 + alpha * alpha) * dt *
                                              (gamma_el * cross(spin[index] + k3[index], B_eff_end + cross(alpha * (spin[index] + k3[index]), B_eff_end)) +
                                               (*v_s_active) * (-1) * (alpha - beta) * cross(spin[index] + k3[index], T_end) +
                                               (*v_s_active) * (-1) * (alpha * beta + 1) * T_end))
                + (-1.0) * (spin[index] + k3[index]);

    // ----------------------------------------------------------------- Calculation of spin at next time step with RK4 -----------------------------------------------------------------

    spin[index] = spin[index] + ((k1[index] + 2.0f * k2[index] + 2.0f * k3[index] + k4) / 6);
  }
}



__global__ void AvgStep(float3 *spin, float *avgTex)
{                                                                                                                           // Programm laeuft ueber verschiedene Threads die parallel auf GPU laufen
  int index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x; // flattened 2D block and grid
  if (0 <= index && index < (size_x * size_y - 1))
  { // Pruefung ob Thread im Bereich des Arrays ist
    avgTex[index] = avgTex[index] + spin[index].z; // z-Projektion wird auf avg-Tex drauf addiert
  }
  
}





__global__ void CalQTopo(float3 *spins, bool *mask, float *results)
{
  int index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x; // flattened 2D block and grid

  // Compute the indices for neighbors
  int index_up = index + 1;  // spins[x][y+1]
  int index_right = index + size_y;  // spins[x+1][y]
  int index_diag = index + size_y + 1;  // spins[x+1][y+1]

  // Check bounds and mask
  if (index < size_x * size_y && index >= 0 && mask[index] && mask[index_up] && mask[index_right] && mask[index_diag])
  {
    float3 r1 = spins[index];
    float3 r2 = spins[index_right];
    float3 r3 = spins[index_up];
    float3 r4 = spins[index_diag];

    results[index] = ComputeTriangleCharge(r1, r2, r3, r4);
  }
}
