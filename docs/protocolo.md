# Protocolo de coleta e avaliação

O experimento usa os 181 arquivos fornecidos, com 176 títulos. As cinco repetições de título compartilham uma classe, mas as duas gravações entram no catálogo. O catálogo contém as faixas inteiras e o modelo combina MFCC com alinhamento de fingerprints. A tarefa é reconhecer essas gravações quando reproduzidas, incluindo partes diferentes da música. Outras edições, covers e apresentações ao vivo exigem novas referências e avaliação.

## Coleta

Comece com um ensaio do modelo já treinado: confira o áudio e teste faixas de vários álbuns, incluindo as mais difíceis de `resultados-por-faixa.md`. Registre quais foram realmente tocadas. Um ensaio com dez músicas não valida os 176 títulos. Para avaliar o catálogo completo, percorra todas as faixas e condições previstas; para uma primeira demonstração, declare a cobertura reduzida.

O treino por pastas aceita as 177 classes, incluindo `ambiente`. Para conservar a saída completa, precisa haver sessões de todas elas em cada divisão; copiar os MP3s para pastas de microfone não substitui essa coleta. Não é necessário retreinar antes do primeiro ensaio do firmware fornecido.

Para cada faixa, escolha três trechos de aproximadamente 30 segundos que representem sua variação: introdução, voz e parte instrumental, quando existirem. Anote a edição utilizada e os intervalos. Uma gravação ao vivo ou um cover não é automaticamente a mesma classe que a gravação de estúdio para este experimento.

Faça pelo menos oito gravações de treino, três de validação e três de teste por classe. O mínimo de duas sessões que o código aceita serve para executar o fluxo, não como garantia de qualidade. Varie a distância (por exemplo, 0,5 m e 1,5 m), o volume e a posição do microfone. Grave o teste em outra sessão, com as condições anotadas. Reserve intervalos que não apareceram nas consultas de treino, mantendo a faixa inteira no catálogo de referência. Isso avalia uma nova consulta a uma gravação conhecida, não generalização para músicas nunca cadastradas.

Não corte a mesma gravação em janelas e distribua aleatoriamente entre treino e teste. Janelas vizinhas são muito parecidas. Separe as sessões antes de extrair as características. O script rejeita arquivos com PCM idêntico, mas esse teste não detecta todos os recortes sobrepostos: a separação correta ainda depende da coleta.

A classe `ambiente` inclui gravações independentes de conversa, ventilador, silêncio, músicas de outros artistas e gravações do próprio Pink Floyd que não estejam no catálogo. Equilibre a duração das classes. Um modelo que nunca viu exemplos negativos pode atribuir confiança alta a uma música desconhecida.

Treine somente com `train`. Escolha os limiares com `val`; o script faz isso por macro-F1 com penalidade para falsos positivos. Consulte `test` só depois de fixar o modelo. Se usar o teste para escolher mudanças, reserve novas sessões para a avaliação final.

## Ensaios na placa

Primeiro faça uma gravação curta no modo `record` e escute o WAV. Confirme que não há áudio zerado, velocidade alterada, estalos ou saturação. Meça clipping e RMS. Quando voltar ao detector, observe os contadores antes de avaliar músicas.

| Ensaio | Duração sugerida | O que registrar |
| --- | --- | --- |
| Cada música em condição favorável | 60 s por faixa | acertos, rejeições, tempo até acender o LED |
| Cada música com conversa e maior distância | 60 s por condição | acurácia, falsos negativos e latência |
| Músicas de outros artistas e gravações fora do catálogo | 2 min | falsos alertas por minuto |
| Silêncio e ruído ambiente | 2 min | falsos alertas, RMS e resets |
| Execução contínua | 10 min | perdas, heap mínimo, stack e reinícios |
| Sobrecarga artificial | 60 s | descartes, quebra de continuidade e recuperação |

Use `tools.bench` em uma pasta nova por ensaio. O parâmetro `--label` é o rótulo verdadeiro do trecho tocado; use `desconhecida` para negativos. Abrir a serial pode reiniciar a placa, como ocorreu com o CP2102 neste Mac. Observe a mensagem de boot e reserve o início da coleta para estabilização; mantenha a conexão aberta durante a reprodução. A acurácia por janela usa `candidate`; o LED inclui o atraso de confirmação e deve ser avaliado separadamente.

Para medir o tempo a partir do início acústico, filme a reprodução e o LED juntos ou use um sinal de referência e um analisador lógico. `software_e2e_us` não mede esse instante físico. Para tempos até a primeira confirmação, use os valores não nulos de `confirmation_us`, que são emitidos quando um alerta começa.

O modo `stress` adiciona 80 ms de espera por frame na tarefa de características, acima do período de 32 ms. Grave-o com `pio run -e stress -t upload --upload-port PORTA` e acompanhe as estatísticas. Espera-se aumento de `ring_drops` e `dsp_deadlines`, continuidade da captura, ausência de confirmação com janelas quebradas e LED apagado depois do timeout. Esses comportamentos precisam ser observados, não presumidos. Restaure `detector` ao terminar.

## Fechar o relatório

Inclua a matriz de confusão do teste real, acurácia, macro-F1 e falsos positivos. Identifique quantidade de gravações, faixas, trechos, distância e ruído. Para a placa, apresente média, p95 e máximo de cada etapa, perdas, memória livre mínima e stack. Não misture resultados do computador com os da placa nem números sintéticos com os de Pink Floyd.

Metas iniciais de engenharia: acurácia acima de 90% no teste separado, nenhum falso alerta nos ensaios negativos de dois minutos, p95 de extração por frame abaixo de 32 ms e zero perdas em dez minutos no modo normal. São metas propostas para orientar os ajustes; o enunciado não estabelece esses valores como corte para nota 10.

## Reprodução no computador

Execute os comandos abaixo na raiz do repositório.

Os áudios ficam fora do Git. Instale FFmpeg e informe a pasta organizada por álbum. Os hashes e nomes estão em `docs/catalogo-local.json`. As músicas negativas têm licença CC BY 4.0, com [créditos e fontes](creditos-audios.md).

O manifesto atual reproduz o experimento publicado, incluindo a duplicação de `no_frills_cumbia.mp3` em treino e teste. Os scripts avisam sobre essa sobreposição. Para uma nova avaliação independente, corrija o manifesto, gere outro conjunto de consultas e repita treino, quantização e testes; os resultados atuais não valem automaticamente para essa correção.

```bash
uv run python -m tools.negatives
uv run python -m tools.catalog --source /caminho/pink-floyd
uv run python -m tools.train --data data/full_catalog/features.npz
uv run python -m tools.quantize --data data/full_catalog/features.npz
uv run python -m tools.evaluate --data data/full_catalog/features.npz \
  --output docs/resultados-pink-floyd.json
uv run python -m tools.ablation --data data/full_catalog/features.npz
uv run python -m tools.playback --source /caminho/pink-floyd
uv run python -m tools.playback --source /caminho/pink-floyd --drop-every 1 \
  --output docs/reproducao-com-perdas.json
uv run pytest -q
pio run -e detector -e detector_fp16 -e detector_int16 -e detector_int8 -e detector_int4
```

`tools.catalog` cadastra todos os arquivos do inventário. Para outra biblioteca, atualize-o antes com `uv run python -m tools.library --source /caminho/pink-floyd`. Este formato aceita até 256 referências, sujeito ao espaço da placa. O catálogo completo usa os rótulos de `model/catalog.json`. O treino exporta ONNX e pesos C; a quantização gera as variantes. Depois de mudar os dados ou treinar, gere as variantes e compile novamente. O compilador rejeita modelos de outro catálogo e modelos sem fingerprints; a confirmação temporal depende dessas referências.

Para desenvolver sem consultar o teste final, use `uv run python -m tools.catalog --source /caminho/pink-floyd --development` e treine com `data/full_catalog/development.npz`. Esse cache contém apenas treino e validação.
