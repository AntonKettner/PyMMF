

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

__device__ float ComputeTriangleCharge(float3 r_1, float3 r_2, float3 r_3) {
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
texture<float, 3, cudaReadModeElementType> v_s; // Textur fuer v_s

__global__ void SimStep(float3 *spin, bool *mask, int *v_s_active)                                                            // mit , float3 *B_tot am Ende wenn log von B_eff gewuenscht
{                                                                                                                           // Programm laeuft ueber verschiedene Threads die parallel auf GPU laufen
  int index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;  // flattened 2D block and grid
  int y     = index % size_y;
  int x     = (index - y) / size_y;

  if (index < size_x * size_y && mask[index]) // Pruefung mit Maske ob Spin simuliert werden muss
  {
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

    // slope at the beginning of the interval using euler
    spin[index] = normalize(spin[index] + (-1.0) / (1.0 + alpha * alpha) * dt *
                                              (gamma_el * cross(spin[index], B_eff + cross(alpha * spin[index], B_eff))
                                              + (*v_s_active) * (-1) * (alpha - beta) * cross(spin[index], T)
                                              + (*v_s_active) * (-1) * (alpha * beta + 1) * T
                                              ));
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
  int index_diag = index_right + 1;  // spins[x+1][y+1]

  // Check bounds and mask
  if (index < size_x * size_y && index >= 0 && mask[index] && mask[index_up] && mask[index_right])
  {
    float3 r1 = spins[index];
    float3 r2 = spins[index_right];
    float3 r3 = spins[index_up];

    float charge1 = ComputeTriangleCharge(r1, r2, r3);

    float charge2 = 0.0f;

    if(!mask[index_diag]) 
    {
      float3 r4 = spins[index_diag];
      charge2 = ComputeTriangleCharge(r1, r2, r4);
    }
    results[index] = charge1 + charge2;
  }
}
