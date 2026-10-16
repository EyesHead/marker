# Marker: PDF → Markdown с бюджетом до $1 на PDF

Форк [EyesHead/marker](https://github.com/EyesHead/marker) для PDF, включая русскоязычные СП и ГОСТ. Запускатель использует исходную разметку Marker и **OpenAI `gpt-6-luna`**. По умолчанию установлен **общий предел $1 на один PDF**; его можно уменьшить через `--budget-usd`. Автоматического перехода на более дорогую модель нет.

Названия `ScriptOllama.py`, `ScriptOllama.sh` и `ScriptMacOS-Ollama.sh` сохранены для совместимости. Облачный провайдер по умолчанию теперь OpenAI. Ollama выбирается явно через `--provider ollama`.

## Результат

- Markdown с HTML-таблицами, сохранёнными строками, заголовками, `rowspan` и `colspan`.
- Все продолжения одной логической таблицы передаются в одном задании модели. Геометрия ячеек и исходные координаты остаются данными PDF.
- Формулы в ячейках — MathML с точным LaTeX в `<annotation encoding="application/x-tex">`. Вне таблиц используется LaTeX в Markdown.
- Рисунки и схемы сохраняются в PNG из исходного PDF. Модель не перерисовывает их.

Локально выполняются извлечение текста, геометрии и лёгкое определение разметки. Большая локальная модель и сервер Ollama не требуются. Сканы и повреждённые области распознаются облачной моделью. Проверки структуры и доступного исходного текста помогают обнаруживать потери, но не доказывают безошибочность распознавания.

## Бюджет и модели

| Модель | Вход за 1 млн токенов | Выход за 1 млн токенов |
|---|---:|---:|
| [`gpt-6-luna`](https://developers.openai.com/api/docs/models/gpt-6-luna) — по умолчанию | $0.10 | $0.50 |
| [`gpt-6.1-sol`](https://developers.openai.com/api/docs/models/gpt-6.1-sol) — явно через `--model` | $2 | $10 |

Тарифы проверены 16 октября 2026 года. Скрытые токены рассуждения входят в оплачиваемый выход. Учёт бюджета учитывает применимые расходы кэширования и резервирует стоимость запроса до генерации. Это предел стоимости токенов модели; налоги и другие начисления аккаунта в него не входят.

Перед генерацией точное число входных токенов, включая изображения, запрашивается через OpenAI Responses. Если посчитать вход или безопасно зарезервировать стоимость невозможно, генерация не начинается. Повторы и исправления ответа используют тот же бюджет; повторы SDK отключены.

Чтобы сократить обращения, пригодный обычный текст извлекается локально, а целая таблица распознаётся одним отдельным заданием. HTML передаётся без JSON-обёртки. Успешные задания сохраняются на диске; совпадающие изображения и промпты внутри работающего процесса используют уже проверенный ответ с повторной проверкой по текущему источнику. Одновременные одинаковые задания разделяют одну генерацию. Размеры рисунков и разрешение основных кропов для этой экономии не уменьшаются. В отчёте бюджета видны `generation_calls`, `token_count_calls` и `validated_cache_hits`.

Журнал расходов хранится в `<кэш Marker>/budgets/` и привязан к SHA-256 содержимого PDF. Повторный запуск, `--force`, переименование файла и смена Luna на Sol используют общий журнал этого PDF при том же каталоге бюджета. Не удаляйте журнал для продолжения: иначе прежние расходы не будут известны запускателю. Понижение `--budget-usd` не возвращает уже потраченные деньги.

При исчерпании $1 новые генерации останавливаются, предыдущая готовая заметка сохраняется. Успешные задания остаются в кэше; отчёт содержит расход и число обращений. Автоматического увеличения предела нет.

## Папки

```text
PolytechBrain/
  ScriptOllama.py
  ScriptOllama.sh            # macOS / Linux
  ScriptMacOS-Ollama.sh     # прежнее имя для macOS
  ScriptOllama.cmd          # Windows
  ScriptOllama.ps1          # PowerShell, необязательно
  PDF/
    document.pdf
  Output/
    document/
      document.md
      document_meta.json
      ...PNG рисунков...
```

`PDF` и `Output` находятся рядом с `ScriptOllama.py`, независимо от текущей папки терминала. Пути с пробелами и кириллицей поддерживаются. Исходные PDF не изменяются. Параметр `--directory` задаёт другую папку с `PDF` и `Output`.

## Установка и ключ

Нужны Python 3.10+, Git, интернет, [API-ключ OpenAI](https://platform.openai.com/api-keys) и доступ к выбранной модели. Ключ запрашивается в окне со скрытым вводом: системный диалог на macOS, Tk на Windows/Linux. Если GUI недоступен, используется скрытый ввод в терминале. Отмена прекращает запуск. Можно заранее задать `OPENAI_API_KEY` в окружении.

Ключ не передаётся в аргументах, не записывается в настройки или результаты и передаётся только процессу конвертации. При OpenAI значение `OLLAMA_API_KEY` не используется.

### macOS

В папке хранилища:

```bash
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.py -o ScriptOllama.py
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.sh -o ScriptOllama.sh
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptMacOS-Ollama.sh -o ScriptMacOS-Ollama.sh
chmod +x ScriptOllama.sh ScriptMacOS-Ollama.sh
./ScriptOllama.sh --file "document.pdf"
```

Прежняя команда `./ScriptMacOS-Ollama.sh` также работает.

### Ubuntu 25

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev git curl build-essential python3-tk
# Далее в папке хранилища:
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.py -o ScriptOllama.py
curl -fL https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.sh -o ScriptOllama.sh
chmod +x ScriptOllama.sh
./ScriptOllama.sh --file "document.pdf"
```

### Windows 10/11

Установите Python 3.10+ с Python Launcher (`py`), Tk и Git. В PowerShell, в папке хранилища:

```powershell
Invoke-WebRequest 'https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.py' -OutFile ScriptOllama.py
Invoke-WebRequest 'https://raw.githubusercontent.com/EyesHead/marker/refs/heads/codex/my-change/ScriptOllama.cmd' -OutFile ScriptOllama.cmd
.\ScriptOllama.cmd --file "document.pdf"
```

## Запуск

Положите PDF в папку `PDF`.

```bash
# Один выбранный файл; Luna, общий бюджет до $1
./ScriptOllama.sh --file "document.pdf"

# Меньший общий предел для этого PDF
./ScriptOllama.sh --file "document.pdf" --budget-usd 0.30

# Явно выбрать Sol с тем же общим журналом расходов
./ScriptOllama.sh --file "document.pdf" --model gpt-6.1-sol --force

# Обновить установленный Marker
./ScriptOllama.sh --update-marker --file "document.pdf"

# Все незавершённые PDF непосредственно в папке PDF
./ScriptOllama.sh
```

`--file` принимает точное имя файла, включая расширение, без пути. Другие версии PDF при этом не обрабатываются. На Windows используются те же параметры после `.\ScriptOllama.cmd`. Готовые заметки пропускаются; `--force` пересобирает выбранную заметку, сохраняя применимые успешные задания и журнал расходов.

`--force-ocr` повторно распознаёт обычный текст; `--disable-ocr` сохраняет доступный текстовый слой, продолжая распознавание обязательной математики и таблиц. Эти параметры несовместимы.

## Явный Ollama

```bash
./ScriptOllama.sh --provider ollama --model kimi-k3 --file "document.pdf" --max-requests 12
```

Этот режим использует `OLLAMA_API_KEY`, `OLLAMA_MODEL` и `OLLAMA_BASE_URL`, а также отдельный скрытый диалог ключа Ollama. `OPENAI_API_KEY` в процесс Ollama не передаётся. Денежный предел OpenAI к этому режиму не применяется: `--max-requests` ограничивает число запросов, включая повторы, но не стоимость Ollama.

## Кэш и параметры

Первый запуск клонирует репозиторий в отдельный кэш, создаёт `venv` и устанавливает Marker. Прежняя установка без поддержки бюджетного OpenAI обновляется однократно. Обычные повторные запуски не клонируют репозиторий заново; `--update-marker` обновляет его явно. Локально изменённый кэш автоматически не перезаписывается.

Кэш: macOS — `~/Library/Caches/eyeshead-marker` (существующий `~/.cache/eyeshead-marker` сохраняется); Linux — `$XDG_CACHE_HOME/eyeshead-marker` или `~/.cache/eyeshead-marker`; Windows — `%LOCALAPPDATA%\eyeshead-marker`. Переопределение: `--cache-dir` / `MARKER_CACHE`. Чтобы сохранялся общий предел расходов, используйте один и тот же каталог кэша.

Успешные задания и исходные PNG для проверки находятся в `Output/.marker-work/`. Ключи в этих каталогах не сохраняются. Распознавание использует исходную разметку Marker; отдельный JSON таблиц, pg-vector и изменения подготовки RAG-батчей не добавляются.

| Параметр | По умолчанию |
|---|---|
| `--provider` | `openai` |
| `--model` | `gpt-6-luna`; `OPENAI_MODEL` для OpenAI, `OLLAMA_MODEL` для Ollama |
| `--base-url` | `https://api.openai.com/v1`; соответствующая `OPENAI_BASE_URL` / `OLLAMA_BASE_URL` |
| `--budget-usd` | `1`; допустимо `0 < значение ≤ 1` |
| `--context-tokens` | `245760` |
| `--max-output-tokens` | `16384`, включая скрытое рассуждение |
| `--max-images` | `16` |
| `--crop-dpi` | `600` |
| `--max-attempts` | `2` для OpenAI, `3` для Ollama; включая первую попытку |
| `--max-concurrency` | `2`, диапазон 1–4 |
| `--max-requests` | `0`, без дополнительного ограничения числа запросов |

Модель получает уменьшенные копии изображений, а сохраняемые рисунки сохраняют исходное разрешение. Контекст — клиентский бюджет, а не настройка сервера. Ни уменьшение бюджета, ни выбор Luna не заменяют визуальную проверку результата.

По умолчанию устанавливается ветка `codex/my-change`. Обновление кэша не заменяет скачанные файлы запускателя: при изменении запускателя их нужно скачать повторно.

Эта миграция подготовлена без платных запросов. Пользовательская проверка реальных PDF выполняется отдельно; полнота и точность нового провайдера на конкретном документе пока не подтверждены.
