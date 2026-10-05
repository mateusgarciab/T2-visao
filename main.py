import sys
import os

from calibracao import calibrarCamera
from projecao_3d import executar_experimento_projecao

def main():

    print("=" * 80)
    print(" PIPELINE COMPLETO DE CALIBRAÇÃO DE CÂMERA (MÉTODO DE ZHANG)")
    print("=" * 80)
    
    # Etapa 1: Calibração Monocular e Undistortion
    print("\n[ETAPA 1/2] Calibração Monocular e Remoção de Distorção (TODAS as imagens)...")
    calibrarCamera()
    
    # Etapa 2: Experimento de Projeção 3D -> 2D
    print("\n[ETAPA 2/2] Experimento de Projeção 3D -> 2D (TODAS as imagens)...")
    executar_experimento_projecao(params_path='resultados/calibracao_parametros.npz', output_dir='resultados')
    
    print("=" * 80)
    print(" PIPELINE EXECUTADO COM SUCESSO! TODOS OS RESULTADOS SALVOS EM 'resultados/'.")
    print("=" * 80)

if __name__ == '__main__':
    main()
