# Meu Tema — Visual Studio Code

Extensão local de tema escuro para o Visual Studio Code, criada a partir do arquivo JSONC fornecido. As propriedades ativas de `colors` e `tokenColors` foram preservadas; comentários e sintaxe específica de JSONC foram removidos para gerar JSON válido.

## Instalação

1. Extraia o arquivo ZIP em uma pasta local.
2. Abra o Visual Studio Code.
3. Pressione `Ctrl + Shift + P`.
4. Execute **Developer: Install Extension from Location...**.
5. Selecione a pasta extraída `meu-tema-vscode` (a pasta que contém `package.json`).
6. Abra **Preferences: Color Theme** e escolha **Meu Tema**.

## Conteúdo

- `package.json`: metadados e declaração do tema.
- `themes/meu-tema-color-theme.json`: cores da interface e realce de sintaxe.

## Observação

Este pacote é para instalação local e não precisa ser publicado no Marketplace.
