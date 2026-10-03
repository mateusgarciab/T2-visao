"""
===============================================================================
Módulo de Experimento de Projeção 3D -> 2D para TODAS as Imagens
Trabalho Acadêmico de Visão Computacional e Processamento de Imagens - VRI/UFPR
===============================================================================
"""

import cv2
import numpy as np
import os

def executar_experimento_projecao(params_path='resultados/calibracao_parametros.npz', output_dir='resultados'):
    """
    Realiza o experimento de projeção 3D para 2D (cv2.projectPoints) para 
    TODAS as imagens válidas da calibração, salvando as imagens com eixos 3D 
    (XYZ) e marcadores para cada foto individual em resultados/.
    """
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Arquivo de parâmetros não encontrado em '{params_path}'. Execute calibracao.py primeiro.")

    data = np.load(params_path, allow_pickle=True)
    K = np.array(data['K'], dtype=np.float32)
    dist = np.array(data['dist'], dtype=np.float32)
    valid_images = data['valid_images']
    objpoints_list = data['objpoints']
    imgpoints_list = data['imgpoints']
    rvecs = data['rvecs']
    tvecs = data['tvecs']

    print(f"=== Experimento de Projeção 3D -> 2D para TODAS as {len(valid_images)} Imagens ===")

    tabela_markdown_global = []
    tabela_markdown_global.append("| Imagem # | Nome do Arquivo | Erro Euclidiano Médio (px) | Erro Mínimo (px) | Erro Máximo (px) | Arquivo 3D Gerado |")
    tabela_markdown_global.append("|:--------:|:---------------:|:-------------------------:|:----------------:|:----------------:|:-----------------:|")

    axis_length = 90.0  # 90 mm (3 quadrados)
    axis_3d = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],       # Eixo X (Vermelho)
        [0, axis_length, 0],       # Eixo Y (Verde)
        [0, 0, -axis_length]       # Eixo Z (Azul)
    ])

    primeira_tabela_md = None

    for idx, img_path in enumerate(valid_images, 1):
        rvec = np.array(rvecs[idx-1], dtype=np.float32)
        tvec = np.array(tvecs[idx-1], dtype=np.float32)
        objpoints = np.array(objpoints_list[idx-1], dtype=np.float32)
        imgpoints_det = np.array(imgpoints_list[idx-1], dtype=np.float32).reshape(-1, 2)

        # 1. Projeta os pontos 3D
        projected_points_2d, _ = cv2.projectPoints(objpoints, rvec, tvec, K, dist)
        projected_points_2d = projected_points_2d.reshape(-1, 2)

        # 2. Distância euclidiana
        erros = np.linalg.norm(imgpoints_det - projected_points_2d, axis=1)
        erro_medio = np.mean(erros)
        erro_min = np.min(erros)
        erro_max = np.max(erros)

        out_img_name = f"projecao_3d_img_{idx:02d}.jpg"
        out_proj_path = os.path.join(output_dir, out_img_name)

        tabela_markdown_global.append(
            f"| Imagem {idx:02d} | `{os.path.basename(str(img_path))}` | {erro_medio:.4f} | {erro_min:.4f} | {erro_max:.4f} | `{out_img_name}` |"
        )

        # Se for a Imagem 1, monta a tabela detalhada de 36 pontos para o relatório
        if idx == 1:
            primeira_tabela_md = []
            primeira_tabela_md.append("| Ponto | Coordenadas 3D (X, Y, Z) [mm] | Cantos Detectados (u, v) [px] | Coordenadas Projetadas (u, v) [px] | Erro Euclidiano [px] |")
            primeira_tabela_md.append("|:-----:|:------------------------------:|:-----------------------------:|:----------------------------------:|:-------------------:|")
            for p in range(len(objpoints)):
                X, Y, Z = objpoints[p]
                u_det, v_det = imgpoints_det[p]
                u_proj, v_proj = projected_points_2d[p]
                primeira_tabela_md.append(
                    f"| P{p+1} | ({X:.1f}, {Y:.1f}, {Z:.1f}) | ({u_det:.2f}, {v_det:.2f}) | ({u_proj:.2f}, {v_proj:.2f}) | {erros[p]:.4f} |"
                )

        # 3. Desenhar visualização
        img_orig = cv2.imread(str(img_path))
        img_vis = img_orig.copy()

        for p in range(len(objpoints)):
            u_det, v_det = int(round(imgpoints_det[p][0])), int(round(imgpoints_det[p][1]))
            u_proj, v_proj = int(round(projected_points_2d[p][0])), int(round(projected_points_2d[p][1]))
            cv2.circle(img_vis, (u_det, v_det), 5, (0, 0, 255), -1)
            cv2.drawMarker(img_vis, (u_proj, v_proj), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)

        # Eixos 3D
        axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, dist)
        axis_2d = axis_2d.reshape(-1, 2).astype(int)

        origin, pt_x, pt_y, pt_z = tuple(axis_2d[0]), tuple(axis_2d[1]), tuple(axis_2d[2]), tuple(axis_2d[3])
        cv2.line(img_vis, origin, pt_x, (0, 0, 255), 4)  # X
        cv2.line(img_vis, origin, pt_y, (0, 255, 0), 4)  # Y
        cv2.line(img_vis, origin, pt_z, (255, 0, 0), 4)  # Z

        cv2.putText(img_vis, "X", pt_x, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(img_vis, "Y", pt_y, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(img_vis, "Z", pt_z, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        cv2.putText(img_vis, f"Imagem #{idx:02d} - Erro Medio: {erro_medio:.2f}px", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imwrite(out_proj_path, img_vis)
        
        # Copia a imagem 1 para experimento_projecao_3d.jpg para manter link no relatorio
        if idx == 1:
            cv2.imwrite(os.path.join(output_dir, 'experimento_projecao_3d.jpg'), img_vis)

        print(f"  [OK] [{idx:02d}] Imagem 3D salva: {out_img_name} (Erro Medio: {erro_medio:.4f} px)")

    print(f"\n[OK] Experimento 3D gerado para TODAS as {len(valid_images)} imagens em '{output_dir}/'.\n")
    return tabela_markdown_global, primeira_tabela_md

if __name__ == '__main__':
    executar_experimento_projecao()
