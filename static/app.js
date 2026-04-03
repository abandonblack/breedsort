const imageInput = document.getElementById('imageInput');
const predictBtn = document.getElementById('predictBtn');
const statusText = document.getElementById('status');
const resultBox = document.getElementById('result');
const preview = document.getElementById('preview');
const breed = document.getElementById('breed');
const confidence = document.getElementById('confidence');
const description = document.getElementById('description');
const top3 = document.getElementById('top3');
const feedbackForm = document.getElementById('feedbackForm');
const feedbackStatus = document.getElementById('feedbackStatus');
const feedbackList = document.getElementById('feedbackList');

predictBtn.addEventListener('click', async () => {
  if (!imageInput.files.length) {
    statusText.textContent = '请先选择图片。';
    return;
  }

  const formData = new FormData();
  formData.append('image', imageInput.files[0]);

  statusText.textContent = '模型识别中，请稍候...';
  resultBox.classList.add('hidden');

  try {
    const res = await fetch('/api/predict', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) {
      statusText.textContent = data.error || '识别服务暂不可用。';
      return;
    }
    preview.src = data.image_url;
    breed.textContent = `识别结果：${data.label}（模型：${data.arch || 'unknown'}）`;
    confidence.textContent = `置信度：${data.confidence}%`;
    description.textContent = `品种介绍：${data.description}`;
    top3.innerHTML = data.top3
      .map((item) => `<li>${item.label}：${(item.score * 100).toFixed(2)}%</li>`)
      .join('');

    resultBox.classList.remove('hidden');
    statusText.textContent = '识别完成 ✅';
  } catch (error) {
    statusText.textContent = '识别失败，请重试。';
  }
});

feedbackForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(feedbackForm);

  try {
    const res = await fetch('/api/feedback', { method: 'POST', body: formData });
    const data = await res.json();
    feedbackStatus.textContent = data.message;
    feedbackForm.reset();
    loadFeedback();
  } catch (error) {
    feedbackStatus.textContent = '反馈提交失败。';
  }
});

async function loadFeedback() {
  const res = await fetch('/api/feedback');
  const data = await res.json();
  feedbackList.innerHTML = data.items
    .map(
      (item) => `
      <div class="feedback-item">
        <strong>${item.nickname}</strong>（${item.rating}分）
        <div>${item.message}</div>
        <small>${item.created_at}</small>
      </div>
    `,
    )
    .join('');
}

loadFeedback();
