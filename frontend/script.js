// Global variables
let currentCommentId = null;
let isClassifying = false;

// DOM elements
const classifyForm = document.getElementById('classifyForm');
const commentTextarea = document.getElementById('commentText');
const classifyBtn = document.getElementById('classifyBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');
const resultsSection = document.getElementById('results');
const resultIcon = document.getElementById('resultIcon');
const resultMessage = document.getElementById('resultMessage');
const resultDetails = document.getElementById('resultDetails');
const feedbackMessage = document.getElementById('feedbackMessage');

// Event listeners
classifyForm.addEventListener('submit', handleClassify);

// Main classification function
async function handleClassify(event) {
  event.preventDefault();

  if (isClassifying) return;

  const text = commentTextarea.value.trim();
  if (!text) {
    alert('Please enter a comment to analyze.');
    return;
  }

  // Show loading state
  setLoadingState(true);
  hideResults();

  try {
    const response = await fetch('/api/classify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: text }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    displayResult(result);
  } catch (error) {
    console.error('Classification error:', error);
    showError('Failed to classify comment. Please try again.');
  } finally {
    setLoadingState(false);
  }
}

// Display classification result
function displayResult(result) {
  currentCommentId = result.id;

  // Set icon and colors based on toxicity
  if (result.is_toxic) {
    resultIcon.textContent = '⚠️';
    resultIcon.style.color = '#dc3545';
    resultMessage.style.color = '#dc3545';
  } else {
    resultIcon.textContent = '✅';
    resultIcon.style.color = '#28a745';
    resultMessage.style.color = '#28a745';
  }

  resultMessage.textContent = result.message;
  resultDetails.textContent = `Confidence: ${(result.confidence * 100).toFixed(
    1
  )}%`;

  // Show results
  resultsSection.style.display = 'block';
  resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Handle user feedback
async function handleFeedback(isCorrect) {
  if (!currentCommentId) return;

  try {
    const response = await fetch('/api/feedback', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        comment_id: currentCommentId,
        is_correct: isCorrect,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();

    // Show feedback message
    feedbackMessage.textContent = result.message;
    feedbackMessage.style.display = 'block';

    // Hide feedback message after 3 seconds
    setTimeout(() => {
      feedbackMessage.style.display = 'none';
    }, 3000);
  } catch (error) {
    console.error('Feedback error:', error);
    showError('Failed to submit feedback. Please try again.');
  }
}

// Utility functions
function setLoadingState(loading) {
  isClassifying = loading;
  classifyBtn.disabled = loading;

  if (loading) {
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-block';
  } else {
    btnText.style.display = 'inline';
    btnLoader.style.display = 'none';
  }
}

function hideResults() {
  resultsSection.style.display = 'none';
  feedbackMessage.style.display = 'none';
}

function showError(message) {
  alert(message); // Simple error handling - could be improved with better UI
}

// Auto-resize textarea
commentTextarea.addEventListener('input', function () {
  this.style.height = 'auto';
  this.style.height = this.scrollHeight + 'px';
});

// Sample comments for testing
const sampleComments = [
  'This is a great article, thanks for sharing!',
  "You're an idiot if you believe this nonsense",
  'I disagree with your opinion but respect your view',
  'This movie was amazing, highly recommend it',
  'Go kill yourself, nobody likes you',
];

// Add sample comment button functionality
document.addEventListener('DOMContentLoaded', function () {
  // Add sample button
  const sampleBtn = document.createElement('button');
  sampleBtn.type = 'button';
  sampleBtn.textContent = '📝 Try Sample Comment';
  sampleBtn.style.marginTop = '10px';
  sampleBtn.style.width = '100%';
  sampleBtn.style.background = 'linear-gradient(135deg, #17a2b8, #6f42c1)';

  sampleBtn.addEventListener('click', function () {
    const randomComment =
      sampleComments[Math.floor(Math.random() * sampleComments.length)];
    commentTextarea.value = randomComment;
    commentTextarea.style.height = 'auto';
    commentTextarea.style.height = commentTextarea.scrollHeight + 'px';
  });

  commentTextarea.parentNode.appendChild(sampleBtn);
});
