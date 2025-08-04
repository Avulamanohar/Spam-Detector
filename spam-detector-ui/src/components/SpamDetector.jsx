import React, { useState, useEffect } from 'react';
import {
  FaEnvelope,
  FaSyncAlt,
  FaMoon,
  FaSun,
  FaCheckCircle,
  FaUpload,
  FaFileCsv,
  FaFileExport,
  FaHistory,
} from 'react-icons/fa';
import ClipLoader from 'react-spinners/ClipLoader';
import './SpamDetector.css';
import * as pdfjsLib from 'pdfjs-dist/webpack';
import mammoth from 'mammoth';

const SpamDetector = () => {
  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [fileName, setFileName] = useState('');
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('message');

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
      setDarkMode(true);
      document.documentElement.classList.add('dark');
    } else {
      setDarkMode(false);
      document.documentElement.classList.remove('dark');
    }

    const stored = localStorage.getItem('spamHistory');
    if (stored) setHistory(JSON.parse(stored));
  }, []);

  useEffect(() => {
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  useEffect(() => {
    localStorage.setItem('spamHistory', JSON.stringify(history));
  }, [history]);

  const handleAnalyze = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch('http://localhost:8000/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input }),
      });
      const data = await response.json();
      setResult(data);

      const newEntry = {
        input,
        ...data,
        timestamp: new Date().toISOString(),
      };

      setHistory((prev) => [newEntry, ...prev.slice(0, 9)]);
    } catch (err) {
      alert('Error analyzing input.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setInput('');
    setResult(null);
    setFileName('');
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setFileName(file.name);
    setActiveTab('message');

    const ext = file.name.split('.').pop().toLowerCase();

    if (ext === 'txt') {
      const reader = new FileReader();
      reader.onload = (event) => setInput(event.target.result);
      reader.readAsText(file);
    } else if (ext === 'pdf') {
      const reader = new FileReader();
      reader.onload = async () => {
        const typedArray = new Uint8Array(reader.result);
        const pdf = await pdfjsLib.getDocument(typedArray).promise;
        let fullText = '';
        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const content = await page.getTextContent();
          const strings = content.items.map((item) => item.str);
          fullText += strings.join(' ') + '\n';
        }
        setInput(fullText);
      };
      reader.readAsArrayBuffer(file);
    } else if (ext === 'docx') {
      const reader = new FileReader();
      reader.onload = async (event) => {
        const arrayBuffer = event.target.result;
        const result = await mammoth.extractRawText({ arrayBuffer });
        setInput(result.value);
      };
      reader.readAsArrayBuffer(file);
    } else if (ext === 'doc') {
      alert('Sorry, .doc files are not supported. Please upload .docx files.');
    } else {
      alert('Unsupported file format. Please upload .txt, .pdf, or .docx files.');
    }
  };

  const exportHistoryJSON = () => {
    const dataStr = JSON.stringify(history, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    downloadFile(url, 'spam-history.json');
  };

  const exportHistoryCSV = () => {
    if (history.length === 0) return;
    const headers = ['Input', 'Verdict', 'Timestamp'];
    const rows = history.map(({ input, verdict, timestamp }) =>
      [`"${input.replace(/"/g, '""')}"`, verdict, timestamp].join(',')
    );
    const csvContent = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    downloadFile(url, 'spam-history.csv');
  };

  const downloadFile = (url, filename) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app-wrapper">
      <header className="app-header">
        <div className="logo-title">
          <h1>
            <FaEnvelope /> Email Safety Checker
          </h1>
        </div>
        <button
          onClick={() => setDarkMode((prev) => !prev)}
          className="dark-toggle"
          aria-label="Toggle dark mode"
        >
          {darkMode ? (
            <>
              <FaSun /> Light
            </>
          ) : (
            <>
              <FaMoon /> Dark
            </>
          )}
        </button>
      </header>

      <main className="app-main two-panel">
        <div className={`content-card left-card ${result ? 'shrink' : ''}`}>
          <div className="tab-bar">
            <button
              className={activeTab === 'message' ? 'tab active' : 'tab'}
              onClick={() => setActiveTab('message')}
            >
              <FaEnvelope /> Message
            </button>
            <button
              className={activeTab === 'file' ? 'tab active' : 'tab'}
              onClick={() => setActiveTab('file')}
            >
              <FaUpload /> File
            </button>
          </div>

          {activeTab === 'message' && (
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Paste your email message..."
              className="text-box"
              rows={6}
              disabled={loading}
            />
          )}

          {activeTab === 'file' && (
            <div className="file-upload-inline">
              <label htmlFor="file-upload" className="upload-label">
                <FaUpload /> Upload File
              </label>
              <input
                id="file-upload"
                type="file"
                accept=".txt, .pdf, .doc, .docx"
                style={{ display: 'none' }}
                onChange={handleFileUpload}
                disabled={loading}
              />
              {fileName && <span className="file-name">{fileName}</span>}
            </div>
          )}

          <div className="button-group">
            <button onClick={handleClear} disabled={loading}>
              <FaSyncAlt /> Clear
            </button>
            <button onClick={handleAnalyze} disabled={loading}>
              <FaCheckCircle /> Analyze
            </button>
          </div>

          <div className="button-group">
            <button onClick={exportHistoryCSV} disabled={!history.length}>
              <FaFileCsv /> Export CSV
            </button>
            <button onClick={exportHistoryJSON} disabled={!history.length}>
              <FaFileExport /> Export JSON
            </button>
          </div>
        </div>

        {result && (
          <div className="content-card right-card">
            <div className="analysis-section">
              {loading ? (
                <div className="loader-container">
                  <ClipLoader color="#2196f3" size={40} />
                </div>
              ) : (
                <div
                  className={`result-box ${
                    result.verdict.toLowerCase() === 'spam'
                      ? 'spam-result'
                      : 'ham-result'
                  }`}
                >
                  <p>
                    This message is <strong>{result.verdict.toUpperCase()}</strong>
                  </p>
                </div>
              )}
            </div>

            <div className="history-section scrollable-history">
              <h3>
                <FaHistory /> History
              </h3>
              <ul>
                {history.map((entry, index) => (
                  <li key={index} title={entry.input}>
                    <span className={entry.verdict === 'spam' ? 'spam' : 'ham'}>
                      [{entry.verdict.toUpperCase()}]
                    </span>{' '}
                    {entry.input.length > 60
                      ? entry.input.slice(0, 60) + '...'
                      : entry.input}
                    <br />
                    <small>{new Date(entry.timestamp).toLocaleString()}</small>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <a href="https://github.com/your-repo" target="_blank" rel="noreferrer">
          View Source on GitHub
        </a>
      </footer>
    </div>
  );
};

export default SpamDetector;
