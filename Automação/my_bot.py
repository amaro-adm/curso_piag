import pyautogui as bot

print("\n --- LOGIN AUTOMÁTICO --- \n")
myUser = "Giovani"
myPassword = "789"

bot.click(1100, 320)
bot.write(myUser)
bot.hotkey("Tab")

bot.write(myPassword)

bot.click(1115, 605) # Entrar






# 
# #importando as dependencias
# import pyautogui as ptg
# import time
# #alterando de janela
# ptg.hotkey("alt", "tab")
# time.sleep(1)

# #posicionar o mouse e clicar no cpf
# ptg.moveTo(2770, 534)
# ptg.click()
# #Escrevendo o cpf
# ptg.write("55555555555")

# #posicionar o mouse e clicar na senha
# ptg.moveTo(2770, 635)
# ptg.click()
# ptg.write("kslghdlaksgd")

# ptg.press("Enter")
# #ptg.displayMousePosition()


# # ctrl + c encerra o programa