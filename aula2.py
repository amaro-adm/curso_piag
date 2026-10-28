import os

nomeArquivo = input('Insira o nome do arquivo:\n')
nome = input('Insira o nome a cadastrar: \n')

localDir = os.listdir()
modo = None

if nomeArquivo in localDir:
    continuar = input('O arquivo já existe neste diretório, deseja adicionar o nome a este arquivo? Y / N').upper()
    if continuar == 'Y':
        modo = 'a'
    if continuar == 'N':
         print(f'Você optou por não adicionar ao arquivo {nomeArquivo}.\n')
         nomeArquivo = input('Insira outro nome para o arquivo:\n')
    else:
        
else:
    modo = 'w'

arquivo = open(nomeArquivo, modo, encoding='utf-8')

arquivo.write(f'\n{nome}')

arquivo.close()