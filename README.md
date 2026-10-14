# Marker: PDF → Markdown с HTML-таблицами

Форк [EyesHead/marker](https://github.com/EyesHead/marker) для русскоязычных PDF, включая СП и ГОСТ. Текст и формулы сохраняются в Markdown, все таблицы — в HTML внутри Markdown. Объединённые ячейки используют `rowspan` и `colspan`, рисунки сохраняются отдельными файлами и вставляются через `<img>` в соответствующие ячейки. Продолжения таблиц между страницами объединяются после проверки LLM.

Основной вариант — **Ollama Cloud + Gemma 4 31B**. Gemma работает в облаке; Marker и, при необходимости, OCR работают на вашем компьютере. Результат удобно проверять в режиме чтения Obsidian. Отдельная JSON-реконструкция таблиц удалена.

Версия с восстановленными HTML-таблицами находится в ветке **`codex/my-change`**. Скрипт по умолчанию устанавливает эту ветку. После её слияния в `master` можно явно задать `MARKER_REF=master` при обновлении.

## Запуск на macOS: все PDF из папки

Нужны Python 3.10 или новее, Git, интернет и [API-ключ Ollama](https://ollama.com/settings/keys).

Папки остаются прежними:

```text
PolytechBrain/
  ScriptMacOS-Ollama.sh
  PDF/
    document.pdf
  Output/
    document/
      document.md
      ...изображения...
```

Получите обновлённый скрипт:

```bash
cd "$HOME/BRAIN_University/PolytechBrain"
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptMacOS-Ollama.sh -o ScriptMacOS-Ollama.sh
chmod +x ScriptMacOS-Ollama.sh
```

Для PDF с читаемым текстовым слоем, в том числе на Mac M1 с ограниченной памятью:

```bash
DISABLE_OCR=1 ./ScriptMacOS-Ollama.sh
```

В начале появится окно для ключа Ollama. Ключ используется только на время запуска и не сохраняется в файл. Если `OLLAMA_API_KEY` уже задан в окружении, окно пропускается.

Скрипт один раз скачивает форк и устанавливает зависимости в `~/.cache/eyeshead-marker`. При обнаружении прежнего JSON-конвейера он однократно обновляет установленный код. Последующие запуски используют кэш; автоматического обновления при каждом запуске нет.

Готовые Markdown пропускаются. Результаты прежнего JSON-конвейера, рядом с которыми есть `document_tables.json`, автоматически пересобираются в HTML. Старый отчёт удаляется после успешного сохранения нового Markdown.

## Повторная конвертация и явное обновление

Пересобрать все PDF, включая уже готовые:

```bash
FORCE=1 DISABLE_OCR=1 ./ScriptMacOS-Ollama.sh
```

Обновить установленный Marker из выбранной ветки и пересобрать PDF:

```bash
UPDATE_MARKER=1 FORCE=1 DISABLE_OCR=1 ./ScriptMacOS-Ollama.sh
```

После слияния HTML-версии в `master`:

```bash
MARKER_REF=master UPDATE_MARKER=1 FORCE=1 DISABLE_OCR=1 ./ScriptMacOS-Ollama.sh
```

## Сканы и повреждённый текстовый слой

Для автоматического определения страниц, которым нужен OCR:

```bash
./ScriptMacOS-Ollama.sh
```

Для принудительного OCR всех страниц:

```bash
FORCE=1 FORCE_OCR=1 ./ScriptMacOS-Ollama.sh
```

OCR использует локальные модели Marker и требует памяти. `DISABLE_OCR=1` подходит только для читаемого текстового слоя; его нельзя совмещать с `FORCE_OCR=1`.

## Один PDF через CLI и Ollama Cloud

Однократная установка:

```bash
git clone --branch codex/my-change https://github.com/EyesHead/marker.git
cd marker
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --editable .
```

Введите ключ скрыто. `OPENAI_API_KEY` здесь содержит ключ **Ollama** для совместимого API:

```bash
export OPENAI_API_KEY="$(python -c 'import getpass; print(getpass.getpass("Ollama API key: "))')"
```

Запуск для PDF с текстовым слоем:

```bash
marker_single document.pdf \
  --output_dir Output \
  --output_format markdown \
  --use_llm \
  --llm_service marker.services.openai.OpenAIService \
  --openai_base_url https://ollama.com/v1 \
  --openai_model gemma4:31b \
  --openai_image_format png \
  --disable_ocr \
  --redo_inline_math \
  --max_concurrency 1 \
  --max_output_tokens 32768 \
  --timeout 180
unset OPENAI_API_KEY
```

Для скана уберите `--disable_ocr`. В прямом облачном API используется имя `gemma4:31b`.

Результат: `Output/document/document.md` и изображения рядом. Проверьте сложные таблицы, формулы и склейку продолжений по исходному PDF: LLM может ошибаться даже при корректном HTML.
