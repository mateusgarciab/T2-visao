"""
===============================================================================
Módulo de Calibração de Câmera Monocular (Método de Zhang)
Trabalho Acadêmico de Visão Computacional e Processamento de Imagens - VRI/UFPR
===============================================================================
"""

import cv2
import numpy as np
import glob
import os

def calibrar_camera(dir_imagens='imagens', grid_size=(4, 9), square_size=30.0, output_dir='resultados'):
    """
    Executa a calibração de câmera pelo Método de Zhang (cv2.calibrateCamera),
    extrai os parâmetros intrínsecos e extrínsecos, calcula o erro RMS e salva 
    imagens de cantos detectados e Undistortion para TODAS as imagens válidas em resultados/.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    extensions = ('*.jpeg', '*.jpg', '*.png')
    images_paths = []
    for ext in extensions:
        images_paths.extend(glob.glob(os.path.join(dir_imagens, ext)))
    images_paths = sorted(images_paths)
    
    if not images_paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada na pasta '{dir_imagens}'.")

    print(f"=== 1. Pré-processamento e Detecção dos Cantos ===")
    print(f"Total de imagens carregadas: {len(images_paths)}")
    print(f"Tamanho da grade interna (GRID_SIZE): {grid_size} (largura x altura em cantos internos)")
    print(f"Tamanho do quadrado do tabuleiro (SQUARE_SIZE): {square_size} mm\n")

    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []
    valid_images = []
    image_shapes = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i, fname in enumerate(images_paths):
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shapes is None:
            image_shapes = gray.shape[::-1]

        # Detecção de cantos
        ret, corners = cv2.findChessboardCorners(
            gray, grid_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if not ret:
            ret, corners = cv2.findChessboardCornersSB(
                gray, grid_size, 
                cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
            )

        if ret:
            corners_subpixel = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_subpixel)
            valid_images.append(fname)
            idx_num = len(valid_images)
            
            # Salvar visualização de cantos para TODAS as imagens válidas
            img_corners = cv2.drawChessboardCorners(img.copy(), grid_size, corners_subpixel, ret)
            out_corners_path = os.path.join(output_dir, f"cantos_detectados_img_{idx_num:02d}.jpg")
            cv2.imwrite(out_corners_path, img_corners)

            # Manter cantos_detectados_1.jpg para compatibilidade
            if idx_num == 1:
                cv2.imwrite(os.path.join(output_dir, "cantos_detectados_1.jpg"), img_corners)
            
            print(f"  [OK] [{idx_num:02d}] {os.path.basename(fname)}: Cantos salvas em {os.path.basename(out_corners_path)}")
        else:
            print(f"  [FAIL] {os.path.basename(fname)}: Falha na detecção dos cantos.")

    print(f"\nTotal de imagens válidas utilizadas na calibração: {len(valid_images)} / {len(images_paths)}")
    if len(valid_images) < 3:
        raise RuntimeError("Número insuficiente de imagens válidas para calibração de câmera.")

    print("\n=== 2. Calibração da Câmera Monocular (Método de Zhang) ===")
    rms_error, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shapes, None, None
    )

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = dist.ravel()[:5]

    print("\n--- MATRIZ INTRÍNSECA DA CÂMERA (K) ---")
    print(f"[[{K[0,0]:10.4f}, {K[0,1]:10.4f}, {K[0,2]:10.4f}],")
    print(f" [{K[1,0]:10.4f}, {K[1,1]:10.4f}, {K[1,2]:10.4f}],")
    print(f" [{K[2,0]:10.4f}, {K[2,1]:10.4f}, {K[2,2]:10.4f}]]")

    print("\n--- PARÂMETROS INTRÍNSECOS ---")
    print(f"Distância Focal fx: {fx:.4f} pixels")
    print(f"Distância Focal fy: {fy:.4f} pixels")
    print(f"Ponto Principal cx: {cx:.4f} pixels")
    print(f"Ponto Principal cy: {cy:.4f} pixels")

    print("\n--- COEFICIENTES DE DISTORÇÃO DA LENTE (D) ---")
    print(f"k1 (Radial 1):   {k1:10.6f}")
    print(f"k2 (Radial 2):   {k2:10.6f}")
    print(f"p1 (Tangencial 1): {p1:10.6f}")
    print(f"p2 (Tangencial 2): {p2:10.6f}")
    print(f"k3 (Radial 3):   {k3:10.6f}")

    print(f"\n--- ERRO DE REPROJEÇÃO MÉDIO (RMS) ---")
    print(f"Erro RMS Reprojection Error: {rms_error:.4f} pixels")

    params_path = os.path.join(output_dir, 'calibracao_parametros.npz')
    np.savez(
        params_path,
        K=K, dist=dist, rms=rms_error,
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
        valid_images=np.array(valid_images),
        objpoints=np.array(objpoints, dtype=object),
        imgpoints=np.array(imgpoints, dtype=object),
        image_shape=image_shapes
    )
    print(f"\n[OK] Parâmetros salvos em: {params_path}")

    print("\n=== 3. Remoção de Distorção (Undistortion de TODAS as Imagens) ===")
    for idx_num, img_path in enumerate(valid_images, 1):
        img_orig = cv2.imread(img_path)
        h, w = img_orig.shape[:2]
        
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        img_undist = cv2.undistort(img_orig, K, dist, None, new_K)
        
        # Desenhar linhas guia horizontais
        step = 100
        for y in range(0, h, step):
            cv2.line(img_orig, (0, y), (w, y), (0, 0, 255), 1)
            cv2.line(img_undist, (0, y), (w, y), (0, 255, 0), 1)

        comparativo = np.hstack((img_orig, img_undist))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparativo, f"Original #{idx_num:02d}", (30, 50), font, 1.2, (0, 0, 255), 3)
        cv2.putText(comparativo, f"Corrigida (Undistorted) #{idx_num:02d}", (w + 30, 50), font, 1.2, (0, 255, 0), 3)
        
        out_comp = os.path.join(output_dir, f"undistort_img_{idx_num:02d}.jpg")
        cv2.imwrite(out_comp, comparativo)

        if idx_num == 1:
            cv2.imwrite(os.path.join(output_dir, "undistort_comparativo_1.jpg"), comparativo)
        elif idx_num == 2:
            cv2.imwrite(os.path.join(output_dir, "undistort_comparativo_2.jpg"), comparativo)
            
        print(f"  [OK] [{idx_num:02d}] Undistort salvo em: {os.path.basename(out_comp)}")

    print("\nProcesso de calibração monocular e desdobramento concluído!\n")
    return K, dist, rms_error, rvecs, tvecs, valid_images, objpoints, imgpoints

if __name__ == '__main__':
    calibrar_camera()
