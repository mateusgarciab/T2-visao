import cv2
import numpy as np
import os

"carrega os parametros salvos em 'params_path'"
def carregarParametros(params_path):
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"Arquivo de parâmetros não encontrado em '{params_path}'. "
            "Execute calibracao.py primeiro.")

    with np.load(params_path, allow_pickle=True) as data:
        return {
            "K": np.asarray(data["K"], dtype=np.float32),
            "dist": np.asarray(data["dist"], dtype=np.float32),
            "valid_images": data["validImages"].copy(),
            "objpoints": data["objPoints"].copy(),
            "imgpoints": data["imgPoints"].copy(),
            "rvecs": data["rvecs"].copy(),
            "tvecs": data["tvecs"].copy(),
        }

def criarEixos3d(axis_length = 90.0):
    return np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, -axis_length],
    ])

"""Converte pontos do sistema 3D para coordenadas de imagem """
def projetarPontos(objpoints, rvec, tvec, K, dist):
    pontos_2d, _ = cv2.projectPoints(objpoints, rvec, tvec, K, dist)
    return pontos_2d.reshape(-1, 2)

"""Calcula erro euclidiano por ponto e estatísticas resumidas """
def calcularErros(pontos_detectados, pontos_projetados):
    erros = np.linalg.norm(pontos_detectados - pontos_projetados, axis=1)
    estatisticas = {
        "medio": float(np.mean(erros)),
        "minimo": float(np.min(erros)),
        "maximo": float(np.max(erros)),
    }
    return erros, estatisticas

"""Adiciona na tabela Markdown o resumo de uma imagem """
def adicionarLinhaTabelaGlobal(tabela, idx, img_path, estatisticas, nome_saida):
    tabela.append(
        f"| Imagem {idx:02d} | `{os.path.basename(str(img_path))}` | "
        f"{estatisticas['medio']:.4f} | {estatisticas['minimo']:.4f} | "
        f"{estatisticas['maximo']:.4f} | `{nome_saida}` |"
    )

def criarTabelaDetalhada(objpoints, detectados, projetados, erros):
    linhas = [
        "| Ponto | Coordenadas 3D (X, Y, Z) [mm] | "
        "Cantos Detectados (u, v) [px] | Coordenadas Projetadas (u, v) [px] | "
        "Erro Euclidiano [px] |",
        "|:-----:|:------------------------------:|:-----------------------------:|"
        ":----------------------------------:|:-------------------:|", ]

    for i, (p3d, det, proj, erro) in enumerate(zip(objpoints, detectados, projetados, erros), start=1):
        X, Y, Z = p3d
        u_det, v_det = det
        u_proj, v_proj = proj
        linhas.append(
            f"| P{i} | ({X:.1f}, {Y:.1f}, {Z:.1f}) | "
            f"({u_det:.2f}, {v_det:.2f}) | "
            f"({u_proj:.2f}, {v_proj:.2f}) | {erro:.4f} |"
        )

    return linhas

def desenharPontos(img, detectados, projetados):
    for det, proj in zip(detectados, projetados):
        u_det, v_det = np.rint(det).astype(int)
        u_proj, v_proj = np.rint(proj).astype(int)

        cv2.circle(img, (u_det, v_det), 5, (0, 0, 255), -1)
        cv2.drawMarker(    img, (u_proj, v_proj), (0, 255, 0),    cv2.MARKER_CROSS, 10, 2)

def desenharEixos3d(img, axis_3d, rvec, tvec, K, dist):
    axis_2d = projetarPontos(axis_3d, rvec, tvec, K, dist).astype(int)
    origem, pt_x, pt_y, pt_z = map(tuple, axis_2d)

    cv2.line(img, origem, pt_x, (0, 0, 255), 4)
    cv2.line(img, origem, pt_y, (0, 255, 0), 4)
    cv2.line(img, origem, pt_z, (255, 0, 0), 4)

    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "X", pt_x, fonte, 1.0, (0, 0, 255), 3)
    cv2.putText(img, "Y", pt_y, fonte, 1.0, (0, 255, 0), 3)
    cv2.putText(img, "Z", pt_z, fonte, 1.0, (255, 0, 0), 3)


def gerarVisualizacao(img_path, idx, detectados, projetados, estatisticas, axis_3d, rvec, tvec, K, dist):
    img = cv2.imread(str(img_path))
    if img is None:
        raise OSError(f"Não foi possível abrir a imagem '{img_path}'.")

    desenharPontos(img, detectados, projetados)
    desenharEixos3d(img, axis_3d, rvec, tvec, K, dist)

    cv2.putText(img, f"Imagem #{idx:02d} - Erro Medio: {estatisticas['medio']:.2f}px", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return img


def executar_experimento_projecao(params_path="resultados/calibracao_parametros.npz",output_dir="resultados",):
    os.makedirs(output_dir, exist_ok=True)
    dados = carregarParametros(params_path)

    K = dados["K"]
    dist = dados["dist"]
    valid_images = dados["valid_images"]
    objpoints_list = dados["objpoints"]
    imgpoints_list = dados["imgpoints"]
    rvecs = dados["rvecs"]
    tvecs = dados["tvecs"]

    total = len(valid_images)
    print(f"=== Experimento de Projeção 3D -> 2D para {total} imagens ===")

    tabela_global = [
        "| Imagem # | Nome do Arquivo | Erro Euclidiano Médio (px) | "
        "Erro Mínimo (px) | Erro Máximo (px) | Arquivo 3D Gerado |",
        "|:--------:|:---------------:|:-------------------------:|:----------------:|:----------------:|:-----------------:|", ]
    tabela_detalhada = None
    axis_3d = criarEixos3d()

    for idx, img_path in enumerate(valid_images, start=1):
        pos = idx - 1
        rvec = np.asarray(rvecs[pos], dtype=np.float32)
        tvec = np.asarray(tvecs[pos], dtype=np.float32)
        objpoints = np.asarray(objpoints_list[pos], dtype=np.float32)
        detectados = np.asarray(imgpoints_list[pos], dtype=np.float32).reshape(-1, 2)

        projetados = projetarPontos(objpoints, rvec, tvec, K, dist)
        erros, estatisticas = calcularErros(detectados, projetados)

        nome_saida = f"projecao_3d_img_{idx:02d}.jpg"
        caminho_saida = os.path.join(output_dir, nome_saida)
        adicionarLinhaTabelaGlobal(tabela_global, idx, img_path, estatisticas, nome_saida)

        if idx == 1:
            tabela_detalhada = criarTabelaDetalhada(objpoints, detectados, projetados, erros)

        visualizacao = gerarVisualizacao(    img_path, idx, detectados, projetados, estatisticas,    axis_3d, rvec, tvec, K, dist)
        if not cv2.imwrite(caminho_saida, visualizacao):
            raise OSError(f"Não foi possível salvar '{caminho_saida}'.")

        if idx == 1:
            cv2.imwrite(os.path.join(output_dir, "experimento_projecao_3d.jpg"),visualizacao,)

        print(f"  [OK] [{idx:02d}] Imagem salva: {nome_saida} " f"(erro médio: {estatisticas['medio']:.4f} px)")

    print(f"\n[OK] Experimento concluído. Resultados em '{output_dir}/'.\n")
    return tabela_global, tabela_detalhada

if __name__ == "__main__":
    executar_experimento_projecao()
