import cv2
import numpy as np
import glob
import os

"Procura e ordena todos os caminhos de imagens disponiveis no arquivo 'imgDir'"
def preproces(imgDir = 'imagens'):
    paths = []
    extensions = ('*.jpeg', '*.jpg', '*.png')
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(imgDir, ext)))
    paths = sorted(paths)
    
    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada na pasta '{imgDir}'.")
    return paths

"Detecta os cantos internos do tabuleiro"
def cornersDetection(imgPaths, gridSize, sqSize, outDir):
    print(f"Total de imagens carregadas: {len(imgPaths)}")
    print(f"Tamanho da grade interna (gridSize): {gridSize} (largura x altura em cantos internos)")
    print(f"Tamanho do quadrado do tabuleiro (sqSize): {sqSize} mm\n")
    auxObj = np.zeros((gridSize[0] * gridSize[1], 3), np.float32)
    auxObj[:, :2] = np.mgrid[0:gridSize[0], 0:gridSize[1]].T.reshape(-1, 2) * sqSize

    objPoints = []
    imgPoints = []
    validImages = []
    imgS = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i, fname in enumerate(imgPaths):
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if imgS is None:
            imgS = gray.shape[::-1]

        # Detecção de cantos
        ret, corners = cv2.findChessboardCorners(
            gray, gridSize, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if not ret:
            ret, corners = cv2.findChessboardCornersSB(
                gray, gridSize, 
                cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
            )

        if ret:
            corners_subpixel = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objPoints.append(auxObj)
            imgPoints.append(corners_subpixel)
            validImages.append(fname)
            idxN = len(validImages)
            
            # Salvar visualização de cantos para TODAS as imagens válidas
            img_corners = cv2.drawChessboardCorners(img.copy(), gridSize, corners_subpixel, ret)
            out_corners_path = os.path.join(outDir, f"cantos_detectados_img_{idxN:02d}.jpg")
            cv2.imwrite(out_corners_path, img_corners)

            # Manter cantos_detectados_1.jpg para compatibilidade
            if idxN == 1:
                cv2.imwrite(os.path.join(outDir, "cantos_detectados_1.jpg"), img_corners)
            
            print(f"  [OK] [{idxN:02d}] {os.path.basename(fname)}: Cantos salvas em {os.path.basename(out_corners_path)}")
        else:
            print(f"  [FAIL] {os.path.basename(fname)}: Falha na detecção dos cantos.")

    print(f"\nTotal de imagens válidas utilizadas na calibração: {len(validImages)} / {len(imgPaths)}")
    if len(validImages) < 3:
        raise RuntimeError("Número insuficiente de imagens válidas para calibração de câmera.")
    return objPoints, imgPoints, imgS, validImages

"Efetua o undistortion e salva as comparações com linhas nas imagens"
def undistort(validImages, K, dist, outDir):
    for idxN, imgPath in enumerate(validImages, 1):
        #imagem original
        imgO = cv2.imread(imgPath)
        h, w = imgO.shape[:2]
        
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        #imagem com undistortion
        imgU = cv2.undistort(imgO, K, dist, None, new_K)
        
        # Desenhar linhas guia horizontais
        step = 100
        for y in range(0, h, step):
            cv2.line(imgO, (0, y), (w, y), (0, 0, 255), 1)
            cv2.line(imgU, (0, y), (w, y), (0, 255, 0), 1)

        comparativo = np.hstack((imgO, imgU))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparativo, f"Original #{idxN:02d}", (30, 50), font, 1.2, (0, 0, 255), 3)
        cv2.putText(comparativo, f"Corrigida (Undistorted) #{idxN:02d}", (w + 30, 50), font, 1.2, (0, 255, 0), 3)
        
        outC = os.path.join(outDir, f"undistort_img_{idxN:02d}.jpg")
        cv2.imwrite(outC, comparativo)

        if idxN == 1:
            cv2.imwrite(os.path.join(outDir, "undistort_comparativo_1.jpg"), comparativo)
        elif idxN == 2:
            cv2.imwrite(os.path.join(outDir, "undistort_comparativo_2.jpg"), comparativo)
            
        print(f"  [OK] [{idxN:02d}] Undistort salvo em: {os.path.basename(outC)}")


#Executa a calibração de câmera
def calibrarCamera(imgDir = 'imagens', gridSize=(4, 9), sqSize=30.0, outDir = 'resultados'):
    os.makedirs(outDir, exist_ok=True)
    imgPaths = []

    print(f"====== 1. Pré-processamento======")
    imgPaths = preproces(imgDir);

    print(f"====== 2. Detecção dos Cantos ======")
    objPoints, imgPoints, imgS, validImages = cornersDetection(imgPaths, gridSize, sqSize, outDir)

    print("\n===== 3. Calibração da Câmera Monocular (Método de Zhang) =====")
    rmsE, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objPoints, imgPoints, imgS, None, None
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
    print(f"Erro RMS Reprojection Error: {rmsE:.4f} pixels")

    parP = os.path.join(outDir, 'calibracao_parametros.npz')
    np.savez(
        parP,
        K=K, dist=dist, rms=rmsE,
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
        validImages=np.array(validImages),
        objPoints=np.array(objPoints, dtype=object),
        imgPoints=np.array(imgPoints, dtype=object),
        image_shape=imgS
    )
    print(f"\n[OK] Parâmetros salvos em: {parP}")

    print("\n===== 4. Remoção de Distorção (Undistortion de TODAS as Imagens) =====")
    undistort(validImages, K, dist, outDir)

    print("\nProcesso de calibração monocular e desdobramento concluído!\n")
    return K, dist, rmsE, rvecs, tvecs, validImages, objPoints, imgPoints

if __name__ == '__main__':
    calibrarCamera()
