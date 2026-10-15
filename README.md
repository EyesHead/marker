# Marker: PDF → Markdown через Ollama Cloud

Форк [EyesHead/marker](https://github.com/EyesHead/marker) для русскоязычных PDF, включая СП и ГОСТ. Адаптивный конвертер использует текст, границы и изображения исходного PDF, а области с отсутствующим или повреждённым текстом распознаёт через **Ollama Cloud**, модель по умолчанию — **`gemma4:31b`**. Локальная Gemma и локальный VLM-сервер для этого пути не требуются.

Результат — Markdown с HTML-таблицами. Формулы внутри ячеек отображаются через MathML с точным LaTeX в `<annotation encoding="application/x-tex">`; вне таблиц используется обычный LaTeX в Markdown. Продолжения одной таблицы собираются из исходных страниц; физические ячейки сохраняют свои координаты, `rowspan` и `colspan`. Рисунки остаются PNG-ресурсами, а изображения целых формул сохраняются для проверки распознавания. Отдельный JSON таблиц не создаётся.

Общий запускатель — `ScriptOllama.py` (только стандартная библиотека Python). Обёртки передают ему параметры. Нужны **Python 3.10 или новее**, **Git**, интернет для установки и запросов к облаку и [API-ключ Ollama](https://ollama.com/settings/keys).

Версия адаптивного конвертера публикуется в ветке **`codex/my-change`**; запускатель устанавливает эту ветку по умолчанию. Переход на `master` выполняется явно после слияния.

## Папки и результаты

Разместите **Python-файл и обёртки вместе**. На macOS нужны все три файла, показанные ниже; на Ubuntu — `ScriptOllama.py` и `ScriptOllama.sh`, на Windows — `ScriptOllama.py` и `ScriptOllama.cmd` (или `ScriptOllama.ps1`).

```text
PolytechBrain/
  ScriptOllama.py
  ScriptOllama.sh
  ScriptMacOS-Ollama.sh
  PDF/
    document.pdf
  Output/
    document/
      document.md
      document_meta.json
      ...изображения...
```

`PDF` и `Output` создаются рядом с `ScriptOllama.py`, независимо от текущей папки терминала. Пути с пробелами и кириллицей поддерживаются. Обрабатываются файлы `.pdf`/`.PDF` непосредственно в `PDF`; исходные файлы не изменяются. `--directory "/другая папка"` задаёт другую папку с `PDF` и `Output`.

`document_meta.json` содержит служебную сводку `cloud_conversion`: `status`, `reasons`, `requests`. При неполном распознавании Markdown и ресурсы сохраняются для просмотра, выводятся причины, а запуск завершается с кодом `1`. Такой PDF повторно обрабатывается при следующем запуске. Полностью готовые результаты пропускаются. Результаты без корректной сводки и прежние результаты с `document_tables.json` пересобираются; старый JSON таблиц удаляется после публикации нового результата.

## macOS

Установите Python 3.10+ и Git, затем загрузите **три** файла в папку хранилища:

```bash
mkdir -p "$HOME/BRAIN_University/PolytechBrain"
cd "$HOME/BRAIN_University/PolytechBrain"
python3 --version
git --version
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.py -o ScriptOllama.py
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.sh -o ScriptOllama.sh
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptMacOS-Ollama.sh -o ScriptMacOS-Ollama.sh
chmod +x ScriptOllama.sh ScriptMacOS-Ollama.sh
mkdir -p PDF
```

Положите PDF в `PDF` и запустите:

```bash
./ScriptMacOS-Ollama.sh
```

Существующее имя `ScriptMacOS-Ollama.sh` сохранено. На Mac появится системное окно со скрытым вводом ключа. При недоступном окне используется защищённый ввод в терминале. Отмена прекращает запуск до установки и конвертации.

## Ubuntu 25

Установите необходимые средства и загрузите **два** файла:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev git curl build-essential
mkdir -p "$HOME/PolytechBrain"
cd "$HOME/PolytechBrain"
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.py -o ScriptOllama.py
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.sh -o ScriptOllama.sh
chmod +x ScriptOllama.sh
mkdir -p PDF
./ScriptOllama.sh
```

Положите PDF в созданную папку `PDF` и повторите последнюю команду. Ключ вводится скрыто в терминале. При запуске без терминала используется скрытый GUI-ввод, если доступен Tk; при отсутствии обоих способов запуск сообщает об этом и возвращает код `1`.

## Windows 10/11

Установите Python 3.10+ с Python Launcher (`py`) или Python в `PATH`, а также Git. Откройте PowerShell и загрузите **два** файла:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\PolytechBrain" | Out-Null
Set-Location "$env:USERPROFILE\PolytechBrain"
py -3 --version
git --version
Invoke-WebRequest 'https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.py' -OutFile ScriptOllama.py
Invoke-WebRequest 'https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.cmd' -OutFile ScriptOllama.cmd
New-Item -ItemType Directory -Force PDF | Out-Null
.\ScriptOllama.cmd
```

Положите PDF в `PDF` и повторите `.\ScriptOllama.cmd`. Ключ вводится скрыто в терминале; без терминала используется окно Tk, если оно доступно. Обёртка выбирает `py -3`, затем `python`. Изолированное окружение использует `venv\Scripts\python.exe`.

Для PowerShell есть отдельная обёртка:

```powershell
Invoke-WebRequest 'https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.ps1' -OutFile ScriptOllama.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\ScriptOllama.ps1 --disable-ocr
```

Можно запускать общий файл напрямую: `py -3 .\ScriptOllama.py --help`.

## Установка, кэш и обновление

При первой конвертации запускатель клонирует репозиторий в `source`, создаёт отдельный `venv` и выполняет `python -m pip install --editable source`. Успешная установленная ревизия записывается в `installed-revision`. Неудачная установка повторяется при следующем запуске. Обычные повторные конвертации используют установленный код без запросов к GitHub; для распознавания через Ollama интернет по-прежнему нужен.

| ОС | Кэш по умолчанию |
| --- | --- |
| macOS | `~/Library/Caches/eyeshead-marker`; существующий `~/.cache/eyeshead-marker` используется на прежнем месте |
| Ubuntu | `$XDG_CACHE_HOME/eyeshead-marker` или `~/.cache/eyeshead-marker` |
| Windows | `%LOCALAPPDATA%\eyeshead-marker` |

`MARKER_CACHE` или `--cache-dir` задаёт другой путь. Кэш прежнего конвертера однократно обновляется до выбранной ветки, если в нём отсутствует адаптивный конвертер или остался JSON-конвейер. Изменённый вручную кэш автоматически не перезаписывается: запуск сообщает его путь.

Принудительная пересборка и явное обновление на macOS/Ubuntu:

```bash
FORCE=1 DISABLE_OCR=1 ./ScriptOllama.sh
UPDATE_MARKER=1 FORCE=1 ./ScriptOllama.sh
MARKER_REF=master UPDATE_MARKER=1 FORCE=1 ./ScriptOllama.sh
```

Последняя команда применяется после слияния адаптивной версии в `master`. На Windows те же действия задаются параметрами:

```powershell
.\ScriptOllama.cmd --force --disable-ocr
.\ScriptOllama.cmd --update-marker --force
.\ScriptOllama.cmd --ref master --update-marker --force
```

Обновление кэша Marker не обновляет скачанные файлы запускателя: для этого повторите команды загрузки из раздела вашей ОС.

## Параметры распознавания

`DISABLE_OCR=1` / `--disable-ocr` отключает общее OCR текста; распознавание обязательных формул сохраняется. Если необходимый текст недоступен, результат будет неполным. Для сканов используйте обычный запуск; `FORCE_OCR=1` / `--force-ocr` принудительно включает распознавание текста. `FORCE_OCR` и `DISABLE_OCR` несовместимы.

| Параметр | Переменная окружения | По умолчанию |
| --- | --- | --- |
| `--model` | `OLLAMA_MODEL` | `gemma4:31b` |
| `--base-url` | `OLLAMA_BASE_URL` | `https://ollama.com/v1` |
| `--context-tokens` | `CLOUD_CONTEXT_TOKENS` | `8192` |
| `--max-output-tokens` | `CLOUD_MAX_OUTPUT_TOKENS` | `2048` |
| `--max-images` | `CLOUD_MAX_IMAGES` | `3` |
| `--crop-dpi` | `CLOUD_CROP_DPI` | `600` |
| `--max-attempts` | `CLOUD_MAX_ATTEMPTS` | `3` (исходный запрос и до двух повторов) |
| `--max-requests` | `CLOUD_MAX_REQUESTS` | `0` (без общего лимита на PDF) |
| `--page-batch-size` | `PAGE_BATCH_SIZE` | `30` (совместимая настройка обработки) |
| `--force` | `FORCE=1` | выключено |
| `--update-marker` | `UPDATE_MARKER=1` | выключено |
| `--ref` | `MARKER_REF` | `codex/my-change` |
| `--repo` | `MARKER_REPO` | `https://github.com/EyesHead/marker.git` |

Бюджеты токенов — консервативные клиентские ограничения, а не точный токенизатор или настройка контекста сервера. Они включают резерв ответа. Для запроса копия изображения ограничивается длинной стороной 1536 пикселей и может уменьшаться дополнительно; сохраняемый ресурс сохраняет полное разрешение. Векторные кропы создаются при заданном DPI, растры сохраняют исходные пиксели. Все этапы используют общий счётчик запросов; повторные попытки входят в `--max-requests`.

Пример запуска с пределом десять облачных запросов на PDF:

```bash
./ScriptOllama.sh --max-requests 10 --context-tokens 8192 --max-output-tokens 2048
```

Ключ передаётся конвертеру только через `OLLAMA_API_KEY` в окружении дочернего процесса. Значение из этой переменной заменяет диалог. Запускатель не принимает ключ в аргументах, не сохраняет его в настройках и скрывает его при выводе сообщений дочернего процесса.

## Автоматические проверки

Офлайн-проверки не вызывают Ollama и не скачивают модели:

```bash
python -m unittest discover -s tests/cloud -v
python -m unittest discover -s tests/html_tables -v
```

Проверки пути Windows на другой ОС не заменяют реальный запуск. CI запускает тесты и обёртки на macOS, Ubuntu и Windows Server 2022/2025; отдельно проверяется контейнер Ubuntu 25.10 с Python 3.11 и 3.13. Эти Windows-задания проверяют выполнение кода и командных обёрток; интерфейс ввода ключа на Windows 10/11 требует отдельной проверки на настольной ОС. Ручной запрос к Ollama выполняется отдельно после интеграции, со скрытым вводом ключа и лимитом десять реальных запросов.
