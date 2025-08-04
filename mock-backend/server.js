// backend.js
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const PORT = 8080;

app.use(cors());
app.use(bodyParser.json());

// Spam keywords used for mock detection
const spamWords = ['free', 'buy now', 'click here', 'win', 'money', 'offer'];

// Analyze input text for spam keywords
function analyzeText(text) {
  const lowercaseMsg = text.toLowerCase();
  const matchedWords = spamWords.filter(word => lowercaseMsg.includes(word));
  const isSpam = matchedWords.length > 0;
  return {
    verdict: isSpam ? 'spam' : 'ham',
    confidence: isSpam ? 0.7 + Math.random() * 0.3 : 0.6 + Math.random() * 0.3,
    keywords: matchedWords,
  };
}

// POST /api/message expects { message: "..." }
app.post('/api/message', (req, res) => {
  const { message } = req.body;
  if (!message || typeof message !== 'string') {
    return res.status(400).json({ error: 'Message is required.' });
  }
  const result = analyzeText(message);
  res.json(result);
});

// POST /api/url expects { url: "..." }
app.post('/api/url', (req, res) => {
  const { url } = req.body;
  if (!url || typeof url !== 'string') {
    return res.status(400).json({ error: 'URL is required.' });
  }
  const result = analyzeText(url);
  res.json(result);
});

// POST /api/subject expects { subject: "..." }
app.post('/api/subject', (req, res) => {
  const { subject } = req.body;
  if (!subject || typeof subject !== 'string') {
    return res.status(400).json({ error: 'Subject is required.' });
  }
  const result = analyzeText(subject);
  res.json(result);
});

app.listen(PORT, () => {
  console.log(`Mock backend running at http://localhost:${PORT}`);
});
