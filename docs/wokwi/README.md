# Circuito no Wokwi

Abra a pasta do projeto no VS Code, com a extensão Wokwi instalada, e clique em `diagram.json`. Se abrir como texto, use **Reopen Editor With... → Wokwi Diagram Editor** no menu da aba. Para editar os fios pelo JSON, escolha **Text Editor**.

Compile o teste de ligações na raiz do projeto:

```bash
pio run -e wiring -t wokwi
```

O comando gera `.pio/build/wiring/wokwi.bin` com bootloader, tabela de partições e aplicativo, além do ELF. O `wokwi.toml` usa essa imagem completa.

Aperte Play. Os LEDs verde e vermelho devem alternar a cada meio segundo, um de cada vez. Não há botão. A mensagem inicial da serial identifica o teste de LEDs.

No firmware `detector`, o verde indica música reconhecida e o vermelho pisca quando faltam janelas de áudio válidas. O teste `wiring` não executa o reconhecimento. Depois de mudar código, recompile; depois de mudar `wokwi.toml`, feche a aba do simulador e inicie outra sessão. Para gravar o teste na placa: `pio run -e wiring -t upload --upload-port PORTA`. Depois use `record` para conferir o microfone e `detector` para reconhecer músicas.

O desenho usa um ESP32 DevKit V1. Os componentes personalizados representam as ligações do INMP441 e do capacitor cerâmico. Não produzem áudio nem simulam filtragem. O [I2S do ESP32 tem suporte parcial no Wokwi](https://docs.wokwi.com/guides/esp32#simulation-features); a captura e a acurácia precisam ser ensaiadas na placa real.

Os arquivos `.chip.wasm` estão incluídos. Para recompilar as representações com o [Wokwi CLI](https://github.com/wokwi/wokwi-cli):

```bash
wokwi-cli chip compile docs/wokwi/inmp441.chip.c -o docs/wokwi/inmp441.chip.wasm
wokwi-cli chip compile docs/wokwi/ceramic.chip.c -o docs/wokwi/ceramic.chip.wasm
wokwi-cli lint .
```

O compilador baixa o SDK e o cabeçalho oficial `wokwi-api.h` no primeiro uso. O cabeçalho baixado fica fora do Git.
