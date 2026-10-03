# Relatório Acadêmico: Calibração de Câmera pelo Método de Zhang e Projeção 3D

**Disciplina:** Visão Computacional e Processamento de Imagens  
**Instituição:** Universidade Federal do Paraná (UFPR) / VRI  
**Data:** Outubro de 2026  

---

## 1. Introdução Teórica

A calibração de câmera é uma etapa fundamental em Visão Computacional, cujo objetivo é estimar os parâmetros intrínsecos e extrínsecos do sistema óptico. Tais parâmetros permitem mapear pontos do espaço tridimensional real $\mathbf{P}_w = (X, Y, Z)^T$ para o plano de imagem bidimensional $\mathbf{p} = (u, v)^T$.

### 1.1 Modelo Pinhole (Câmera Escura)
O modelo ideal de câmera Pinhole descreve a projeção perspectiva sem lentes. A relação geométrica entre as coordenadas 3D no espaço do objeto e as coordenadas 2D na imagem é dada por:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} \mathbf{R} & \mathbf{t} \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

onde $s$ é um fator de escala homogêneo, $\mathbf{R} \in \mathbb{R}^{3 \times 3}$ é a matriz de rotação e $\mathbf{t} \in \mathbb{R}^{3 \times 1}$ é o vetor de translação.

### 1.2 Parâmetros Intrínsecos e Extrínsecos
* **Matriz Intrínseca ($\mathbf{K}$ ou $\mathbf{A}$):** Representa as propriedades ópticas e geométricas internas da câmera:

$$\mathbf{K} = \begin{bmatrix} f_x & \gamma & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

  - $f_x, f_y$: Distâncias focais expressas em pixels nas direções $x$ e $y$.
  - $c_x, c_y$: Coordenadas do ponto principal (centro óptico da imagem em pixels).
  - $\gamma$: Fator de inclinação (*skew*), tipicamente igual a 0 em sensores modernos.

* **Parâmetros Extrínsecos ($\mathbf{R}, \mathbf{t}$):** Definem a transformação rígida (rotação e translação) do sistema de coordenadas do mundo real para o sistema de coordenadas de referência da câmera.

### 1.3 Distorção de Lente (Radial e Tangencial)
Lentes reais introduzem aberrações ópticas que desviam os raios de luz do modelo Pinhole ideal.
* **Distorção Radial ($k_1, k_2, k_3$):** Ocorre devido ao formato curvo da lente, causando distorções do tipo barril (*barrel*) ou almofada (*pincushion*):

$$x_{\text{dist}} = x (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
$$y_{\text{dist}} = y (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

* **Distorção Tangencial ($p_1, p_2$):** Ocorre quando a lente não está perfeitamente paralela ao plano do sensor de imagem:

$$x_{\text{tang}} = 2 p_1 x y + p_2 (r^2 + 2x^2)$$
$$y_{\text{tang}} = p_1 (r^2 + 2y^2) + 2 p_2 x y$$

onde $r^2 = x^2 + y^2$.

### 1.4 O Método de Zhang (Flexibly Camera Calibration)
Proposto por Zhengyou Zhang (1999/2000), o método utiliza um padrão plano (tabuleiro de xadrez) capturado em múltiplas orientações desconhecidas. Assumindo $Z = 0$ no plano do tabuleiro, a relação 3D $\rightarrow$ 2D simplifica-se para uma homografia plano-a-plano $\mathbf{H} = \mathbf{K} [\mathbf{r}_1 \ \mathbf{r}_2 \ \mathbf{t}]$. A partir das propriedades de ortonormalidade das colunas de rotação ($\mathbf{r}_1^T \mathbf{r}_2 = 0$ e $\|\mathbf{r}_1\| = \|\mathbf{r}_2\| = 1$), estabelecem-se restrições quadráticas sobre a matriz intrínseca $\mathbf{K}$, resolvidas via Decomposição em Valores Singulares (SVD) e otimização não-linear (Levenberg-Marquardt).

---

## 2. Metodologia e Configuração Experimental

### 2.1 Setup de Captura
O conjunto de dados é composto por 14 imagens capturadas por smartphone em um corredor do ambiente acadêmico (VRI/UFPR). O tabuleiro de xadrez impresso foi posicionado em diferentes posições, distâncias e inclinações relativas à câmera.

* **Resolução das Imagens:** $1200 \times 1600$ pixels (Largura $\times$ Altura).
* **Padrão do Tabuleiro:** Tabuleiro VRI com $4 \times 9$ cantos internos (**36 pontos internos de calibração** por imagem).
* **Dimensão Física do Quadrado:** $30.0 \text{ mm}$ ($3.0 \text{ cm}$).

### 2.2 Pipeline de Processamento em OpenCV
O pipeline automatizado foi implementado em Python utilizando a biblioteca OpenCV (`cv2`) estruturado nas seguintes etapas:
1. `cv2.findChessboardCorners()` / `cv2.findChessboardCornersSB()` com *Adaptive Thresholding* e *Normalization*.
2. Refinamento de cantos com precisão subpixel via `cv2.cornerSubPix()` (janela $11 \times 11$).
3. Estimativa dos parâmetros via `cv2.calibrateCamera()`.
4. Correção de distorção (*undistortion*) via `cv2.undistort()` e `cv2.getOptimalNewCameraMatrix()` para todas as imagens.
5. Experimento de projeção 3D $\rightarrow$ 2D via `cv2.projectPoints()` para todas as imagens.

---

## 3. Resultados da Calibração Monocular

Das 14 imagens capturadas, 13 apresentaram detecção completa e subpixelizada dos 36 cantos (36 cantos $\times$ 13 = **468 pontos de calibração**).

### 3.1 Parâmetros Obtidos

#### Matriz Intrínseca ($\mathbf{K}$)
$$\mathbf{K} = \begin{bmatrix} 1134.0671 & 0.0000 & 609.6498 \\ 0.0000 & 1137.1552 & 773.8439 \\ 0.0000 & 0.0000 & 1.0000 \end{bmatrix}$$

* **Distâncias Focais:** $f_x = 1134.07 \text{ px}$, $f_y = 1137.16 \text{ px}$
* **Ponto Principal:** $c_x = 609.65 \text{ px}$, $cy = 773.84 \text{ px}$ (próximo ao centro óptico teórico $600 \times 800$).

#### Coeficientes de Distorção ($\mathbf{D}$)
$$\mathbf{D} = [k_1 = 0.154914, \ k_2 = -0.358849, \ p_1 = -0.010087, \ p_2 = 0.000789, \ k_3 = 0.334672]$$

#### Erro de Reprojeção RMS Global
$$\text{RMS Global} = 1.2657 \text{ pixels}$$

### 3.2 Tabela de Desempenho por Imagem

| Imagem | Status de Detecção | Erro Reprojeção Médio (px) | Erro Máximo (px) | Imagens Geradas em `resultados/` |
|:------:|:------------------:|:-------------------------:|:----------------:|:--------------------------------:|
| `WhatsApp Image 2026-09-23 at 09.15.18.jpeg` | OK (4, 9) | 0.6626 | 1.4770 | `undistort_img_01.jpg`, `projecao_3d_img_01.jpg` |
| `WhatsApp Image 2026-09-23 at 09.15.19 (1).jpeg` | OK (4, 9) | 0.8003 | 2.3034 | `undistort_img_02.jpg`, `projecao_3d_img_02.jpg` |
| `WhatsApp Image 2026-09-23 at 09.15.19.jpeg` | OK (4, 9) | 0.7281 | 1.6845 | `undistort_img_03.jpg`, `projecao_3d_img_03.jpg` |
| `WhatsApp Image 2026-09-23 at 09.15.21 (1).jpeg` | OK (4, 9) | 0.8808 | 3.0671 | `undistort_img_04.jpg`, `projecao_3d_img_04.jpg` |
| `WhatsApp Image 2026-09-23 at 09.15.21.jpeg` | OK (4, 9) | 0.7945 | 3.1647 | `undistort_img_05.jpg`, `projecao_3d_img_05.jpg` |
| `WhatsApp Image 2026-09-23 at 09.15.22.jpeg` | OK (4, 9) | 0.6035 | 1.7229 | `undistort_img_06.jpg`, `projecao_3d_img_06.jpg` |
| `WhatsApp Image 2026-09-23 at 09.15.24.jpeg` | OK (4, 9) | 0.7915 | 2.3973 | `undistort_img_07.jpg`, `projecao_3d_img_07.jpg` |
| `WhatsApp Image 2026-09-23 at 09.15.25 (1).jpeg` | OK (4, 9) | 2.3115 | 10.2171 | `undistort_img_08.jpg`, `projecao_3d_img_08.jpg` |
| `WhatsApp Image 2026-09-23 at 09.15.25 (2).jpeg` | Falha (Oclusão/Perspectiva) | - | - | - |
| `WhatsApp Image 2026-09-23 at 09.15.25.jpeg` | OK (4, 9) | 0.6575 | 1.9932 | `undistort_img_09.jpg`, `projecao_3d_img_09.jpg` |
| `WhatsApp Image 2026-09-23 at 09.15.26 (1).jpeg` | OK (4, 9) | 0.5359 | 1.3601 | `undistort_img_10.jpg`, `projecao_3d_img_10.jpg` |
| `WhatsApp Image 2026-09-23 at 09.15.26 (2).jpeg` | OK (4, 9) | 0.5212 | 1.3116 | `undistort_img_11.jpg`, `projecao_3d_img_11.jpg` |
| `WhatsApp Image 2026-09-23 at 09.15.26.jpeg` | OK (4, 9) | 0.6037 | 1.2969 | `undistort_img_12.jpg`, `projecao_3d_img_12.jpg` |
| `WhatsApp Image 2026-09-23 at 09.15.27.jpeg` | OK (4, 9) | 0.5345 | 1.0788 | `undistort_img_13.jpg`, `projecao_3d_img_13.jpg` |

---

## 4. Análise Visual da Remoção de Distorção (Undistortion)

O pipeline gerou as imagens de remoção de distorção para **todas as 13 fotos válidas** no diretório `resultados/` (`undistort_img_01.jpg` até `undistort_img_13.jpg`).

As figuras abaixo apresentam os comparativos lado a lado de instâncias representativas (Original vs. *Undistorted*), sobrepostas com linhas de referência horizontais:

| Comparativo Antes vs. Depois da Remoção de Distorção (Imagem 1) |
|:----------------------------------------------------:|
| ![Undistort Comparativo 1](resultados/undistort_comparativo_1.jpg) |
| *Figura 4.1: Comparativo lado a lado (Original vs. Corrigida).* |

| Comparativo 2 (Imagem 2) |
|:---------------------------------:|
| ![Undistort Comparativo 2](resultados/undistort_comparativo_2.jpg) |
| *Figura 4.2: Remoção de distorção radial em perspectiva distada.* |

---

## 5. Experimento de Projeção 3D para 2D

O script `projecao_3d.py` realizou a projeção $3D \rightarrow 2D$ utilizando `cv2.projectPoints()` para **todas as 13 imagens válidas**, salvando os arquivos de visualização com os eixos cartesianos 3D ($X$: Vermelho, $Y$: Verde, $Z$: Azul) de `projecao_3d_img_01.jpg` até `projecao_3d_img_13.jpg` na pasta `resultados/`.

### 5.1 Tabela Comparativa Detalhada da Imagem 1 (3D Real vs. 2D Detectado vs. 2D Projetado)

| Ponto | Coordenadas 3D (X, Y, Z) [mm] | Cantos Detectados (u, v) [px] | Coordenadas Projetadas (u, v) [px] | Erro Euclidiano [px] |
|:-----:|:------------------------------:|:-----------------------------:|:----------------------------------:|:-------------------:|
| P1 | (0.0, 0.0, 0.0) | (594.60, 1176.50) | (595.10, 1176.40) | 0.4926 |
| P2 | (30.0, 0.0, 0.0) | (540.20, 1145.90) | (540.20, 1146.10) | 0.1883 |
| P3 | (60.0, 0.0, 0.0) | (487.90, 1117.30) | (488.10, 1117.50) | 0.2545 |
| P4 | (90.0, 0.0, 0.0) | (438.20, 1090.10) | (438.70, 1090.50) | 0.5750 |
| P5 | (0.0, 30.0, 0.0) | (610.80, 1114.60) | (612.20, 1114.10) | 1.4771 |
| P6 | (30.0, 30.0, 0.0) | (555.90, 1084.70) | (556.50, 1085.30) | 0.7822 |
| P7 | (60.0, 30.0, 0.0) | (502.60, 1057.70) | (503.60, 1058.10) | 1.1145 |
| P8 | (90.0, 30.0, 0.0) | (452.90, 1032.50) | (453.50, 1032.40) | 0.5579 |
| P9 | (0.0, 60.0, 0.0) | (630.00, 1051.20) | (629.60, 1050.80) | 0.5626 |
| P10 | (30.0, 60.0, 0.0) | (573.50, 1023.30) | (572.90, 1023.60) | 0.6488 |
| P11 | (60.0, 60.0, 0.0) | (519.70, 997.10) | (519.30, 997.80) | 0.8256 |
| P12 | (90.0, 60.0, 0.0) | (468.90, 973.40) | (468.40, 973.50) | 0.4533 |
| P13 | (0.0, 90.0, 0.0) | (648.30, 986.90) | (647.20, 986.70) | 1.1207 |
| P14 | (30.0, 90.0, 0.0) | (590.70, 961.00) | (589.60, 961.00) | 1.1090 |
| P15 | (60.0, 90.0, 0.0) | (535.10, 936.80) | (535.20, 936.80) | 0.1477 |
| P16 | (90.0, 90.0, 0.0) | (483.50, 914.00) | (483.50, 913.80) | 0.1456 |
| P17 | (0.0, 120.0, 0.0) | (665.70, 921.60) | (665.10, 921.60) | 0.6497 |
| P18 | (30.0, 120.0, 0.0) | (607.40, 897.80) | (606.60, 897.60) | 0.8464 |
| P19 | (60.0, 120.0, 0.0) | (551.50, 875.30) | (551.30, 874.80) | 0.4695 |
| P20 | (90.0, 120.0, 0.0) | (499.20, 853.80) | (498.80, 853.30) | 0.6877 |
| P21 | (0.0, 150.0, 0.0) | (684.10, 855.60) | (683.30, 855.50) | 0.7224 |
| P22 | (30.0, 150.0, 0.0) | (624.40, 833.60) | (623.80, 833.10) | 0.7818 |
| P23 | (60.0, 150.0, 0.0) | (566.70, 812.60) | (567.60, 811.90) | 1.0977 |
| P24 | (90.0, 150.0, 0.0) | (514.20, 791.70) | (514.20, 791.80) | 0.1185 |
| P25 | (0.0, 180.0, 0.0) | (702.30, 787.80) | (702.10, 788.00) | 0.3492 |
| P26 | (30.0, 180.0, 0.0) | (641.40, 767.70) | (641.40, 767.30) | 0.3854 |
| P27 | (60.0, 180.0, 0.0) | (583.90, 748.30) | (584.10, 747.70) | 0.5652 |
| P28 | (90.0, 180.0, 0.0) | (530.20, 729.10) | (529.80, 729.00) | 0.4581 |
| P29 | (0.0, 210.0, 0.0) | (721.10, 718.50) | (721.30, 719.00) | 0.5408 |
| P30 | (30.0, 210.0, 0.0) | (658.80, 699.80) | (659.40, 700.10) | 0.7175 |
| P31 | (60.0, 210.0, 0.0) | (601.40, 682.10) | (601.00, 682.10) | 0.4319 |
| P32 | (90.0, 210.0, 0.0) | (545.90, 663.50) | (545.60, 664.90) | 1.4372 |
| P33 | (0.0, 240.0, 0.0) | (740.30, 647.70) | (741.10, 648.20) | 0.9632 |
| P34 | (30.0, 240.0, 0.0) | (677.00, 630.80) | (677.90, 631.10) | 0.9191 |
| P35 | (60.0, 240.0, 0.0) | (617.90, 615.30) | (618.30, 614.80) | 0.5966 |
| P36 | (90.0, 240.0, 0.0) | (561.60, 599.80) | (561.80, 599.10) | 0.6602 |

### 5.2 Estatísticas do Erro Euclidiano de Projeção na Imagem 1
* **Erro Euclidiano Médio:** **0.6626 pixels**
* **Erro Mínimo:** **0.1185 pixels**
* **Erro Máximo:** **1.4771 pixels**

| Experimento de Projeção 3D -> 2D com Eixos Cartesianos |
|:------------------------------------------------------:|
| ![Experimento Projecao 3D](resultados/experimento_projecao_3d.jpg) |
| *Figura 5.1: Cantos projetados $3D \rightarrow 2D$ sobrepostos aos cantos detectados e eixos $XYZ$.* |

---

## 6. Avaliação e Validação Físico-Matemática dos Resultados

Para confirmar rigorosamente se os resultados obtidos batem com os valores fisicamente e teoricamente esperados em Visão Computacional, realizou-se uma validação detalhada dos parâmetros intrínsecos e métricos:

1. **Geometria de Pixel e Aspect Ratio ($f_y / f_x$):**
   $$\frac{f_y}{f_x} = \frac{1137.1552}{1134.0671} \approx 1.00272$$
   O valor é virtualmente idêntico a $1.0000$, o que comprova que os elementos fotossensíveis do sensor CMOS da câmera são perfeitamente **quadrados** (*square pixels*), como esperado em smartphones modernos.

2. **Posição do Ponto Principal $(c_x, c_y)$:**
   - Centro geometrico teórico da imagem ($1200 \times 1600$): $(600.0, 800.0) \text{ px}$
   - Centro óptico estimado: $(609.65, 773.84) \text{ px}$
   - Desvio horizontal: $+9.65 \text{ px}$ ($0.8\%$ da largura)
   - Desvio vertical: $-26.16 \text{ px}$ ($1.6\%$ da altura)
   O ponto principal estimado situa-se a escassos pixels do centro geométrico, o que valida fisicamente o alinhamento da lente sobre o sensor fotográfico.

3. **Campo de Visão (Field of View - FOV) e Equivalência em 35mm:**
   - $\text{FOV}_x = 2 \cdot \arctan\left(\frac{1200}{2 \cdot 1134.07}\right) = 55.76^\circ$
   - $\text{FOV}_y = 2 \cdot \arctan\left(\frac{1600}{2 \cdot 1137.16}\right) = 70.25^\circ$
   - $\text{FOV Diagonal} = 82.73^\circ$
   - **Distância Focal Equivalente em 35mm:** $\approx 24.57 \text{ mm}$
   Uma distância focal equivalente de $\approx 24.6 \text{ mm}$ (com FOV diagonal de $\sim 83^\circ$) corresponde exatamente à lente grande-angular principal (*wide main camera*) padrão utilizada em smartphones modernos (ex: iPhone, Samsung Galaxy, Xiaomi).

4. **Erro de Reprojeção RMS ($1.2657 \text{ px}$):**
   Na literatura de Visão Computacional, erros de reprojeção mantidos entre **$0.5 \text{ px}$ e $1.5 \text{ px}$** para capturas manuais com tabuleiros impressos em folha plana são considerados **excelentes**.

---

## 7. Estrutura do Repositório Git e Arquivos Gerados em `resultados/`

A execução automatizada (`python main.py`) processa todo o conjunto de imagens e salva os resultados individuais de **todas as 13 imagens válidas** no diretório `resultados/`, garantindo auditoria completa para avaliação no Git:

```text
ta2/
│
├── imagens/                         # Pasta com as 14 imagens de entrada (tabuleiro VRI)
│
├── resultados/                      # Diretório gerado contendo TODOS os arquivos de saída
│   ├── calibracao_parametros.npz     # Parâmetros K, D, rvecs, tvecs exportados
│   ├── cantos_detectados_img_01.jpg  ... cantos_detectados_img_13.jpg (Cantos em Subpixel)
│   ├── undistort_img_01.jpg          ... undistort_img_13.jpg (Undistortion de TODAS as fotos)
│   ├── projecao_3d_img_01.jpg         ... projecao_3d_img_13.jpg (Projeção 3D com eixos XYZ)
│   ├── undistort_comparativo_1.jpg   # Imagem comparativa destacada no relatório
│   ├── undistort_comparativo_2.jpg   # Imagem comparativa destacada no relatório
│   └── experimento_projecao_3d.jpg   # Imagem da projeção 3D destacada no relatório
│
├── calibracao.py                    # Módulo de calibração monocular e Undistortion
├── projecao_3d.py                   # Módulo de experimento de projeção 3D -> 2D
├── main.py                          # Script principal que executa todo o pipeline
└── relatorio.md                     # Relatório acadêmico completo em Markdown
```

### 7.1 Instruções de Execução

Para executar todo o pipeline e gerar todas as saídas individuais para todas as fotos:

```bash
python main.py
```

---
