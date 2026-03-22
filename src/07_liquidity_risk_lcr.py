import pandas as pd

def validate_lcr_model():
    hqla = 50000000  
    passivos = pd.DataFrame({
        'saldo_atual': [100000000, 50000000, 80000000, 40000000],
        'fator_estresse': [0.05, 0.10, 0.25, 0.40] 
    })
    total_saidas = (passivos['saldo_atual'] * passivos['fator_estresse']).sum()
    entradas_projetadas = 15000000 * 0.50 
    
    lcr = hqla / (total_saidas - entradas_projetadas)
    print(f"LCR Calculado: {lcr * 100:.2f}%")

if __name__ == "__main__":
    validate_lcr_model()
