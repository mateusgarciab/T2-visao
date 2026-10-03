"""
===============================================================================
Pipeline Completo de Calibração de Câmera e Projeção 3D
Trabalho Acadêmico de Visão Computacional e Processamento de Imagens - VRI/UFPR
===============================================================================
"""

import sys
import os

from calibracao import calibrar_camera
from projecao_3d import executar_experimento_projecao

def main():
    print("=" * 80)
    print(" PIPELINE COMPLETO DE CALIBRAÇÃO DE CÂMERA (MÉTODO DE ZHANG)")
    print("=" * 80)
    
    # Etapa 1: Calibração Monocular e Undistortion de TODAS as imagens
    print("\n[ETAPA 1/2] Calibração Monocular e Remoção de Distorção (TODAS as imagens)...")
    calibrar_camera(dir_imagens='imagens', grid_size=(4, 9), square_size=30.0, output_dir='resultados')
    
    # Etapa 2: Experimento de Projeção 3D -> 2D de TODAS as imagens
    print("\n[ETAPA 2/2] Experimento de Projeção 3D -> 2D (TODAS as imagens)...")
    executar_experimento_projecao(params_path='resultados/calibracao_parametros.npz', output_dir='resultados')
    
    print("=" * 80)
    print(" PIPELINE EXECUTADO COM SUCESSO! TODOS OS RESULTADOS SALVOS EM 'resultados/'.")
    print("=" * 80)

if __name__ == '__main__':
    main()
