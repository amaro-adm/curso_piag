lista = ['José', 'João', 'Josefa']

nome = input('Digite o nome para checar se há na lista: \n')

try:
    if nome in lista:
        print('O nome está na lista')
    else:
        raise Exception
except Exception:
    print('O nome não está na lista')