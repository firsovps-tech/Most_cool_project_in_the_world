const form = document.querySelector('#searchForm');
const queryInput = document.querySelector('#queryInput');
const topKSelect = document.querySelector('#topKSelect');
const resultsNode = document.querySelector('#results');
const statusBox = document.querySelector('#statusBox');
const summaryBox = document.querySelector('#summaryBox');
const domainText = document.querySelector('#domainText');
const correctedText = document.querySelector('#correctedText');

function setStatus(message, type = 'info') {
  statusBox.hidden = !message;
  statusBox.textContent = message || '';
  statusBox.className = `status status--${type}`;
}

function clearResults() {
  resultsNode.innerHTML = '';
  summaryBox.hidden = true;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function formatScore(score) {
  if (score === null || score === undefined || Number.isNaN(Number(score))) {
    return '—';
  }

  return Number(score).toFixed(4);
}

function renderResults(data) {
  const prepared = data.prepared_query || {};
  domainText.textContent = prepared.domain || '—';
  correctedText.textContent = prepared.corrected_query_text || data.query || '—';
  summaryBox.hidden = false;

  if (!data.results || data.results.length === 0) {
    resultsNode.innerHTML = `
      <article class="empty-card">
        <h2>Ничего не найдено</h2>
        <p>Попробуйте изменить формулировку запроса или указать больше признаков.</p>
      </article>
    `;
    return;
  }

  resultsNode.innerHTML = data.results.map((item, index) => {
    const characteristics = (item.characteristics || [])
      .slice(0, 8)
      .map((row) => `
        <span class="chip">
          <b>${escapeHtml(row.name)}:</b> ${escapeHtml(row.value)}
        </span>
      `)
      .join('');

    const price = item.price
      ? `<span class="price">${escapeHtml(item.price)} ₽</span>`
      : '<span class="price price--muted">цена не указана</span>';

    return `
      <article class="result-card">
        <div class="result-card__top">
          <span class="rank">#${index + 1}</span>
          <span class="score">score ${formatScore(item.score)}</span>
        </div>
        <h2>${escapeHtml(item.title)}</h2>
        <p>${escapeHtml(item.description)}</p>
        <div class="meta">
          <span>ID: ${escapeHtml(item.id)}</span>
          ${item.article ? `<span>Артикул: ${escapeHtml(item.article)}</span>` : ''}
          ${price}
        </div>
        <div class="chips">${characteristics}</div>
      </article>
    `;
  }).join('');
}

async function runSearch(query, topK) {
  clearResults();
  setStatus('Загружаю модель и ищу подходящие варианты. Первый запуск может занять дольше...', 'info');

  try {
    const response = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: topK }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Ошибка поиска');
    }

    setStatus(`Найдено результатов: ${data.count}`, 'success');
    renderResults(data);
  } catch (error) {
    clearResults();
    setStatus(error.message, 'error');
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();

  if (!query) {
    setStatus('Введите запрос', 'error');
    return;
  }

  runSearch(query, Number(topKSelect.value));
});

document.querySelectorAll('[data-query]').forEach((button) => {
  button.addEventListener('click', () => {
    queryInput.value = button.dataset.query;
    runSearch(button.dataset.query, Number(topKSelect.value));
  });
});
