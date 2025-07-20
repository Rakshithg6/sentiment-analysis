import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    setResult(null);
    try {
      const res = await axios.post('http://localhost:8000/predict', { text });
      setResult(res.data);
    } catch (err) {
      setResult({ label: 'error', score: 0 });
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 500, margin: '60px auto', fontFamily: 'sans-serif' }}>
      <h2>Electronix AI Sentiment Analysis</h2>
      <textarea
        rows={6}
        style={{ width: '100%', fontSize: 16 }}
        placeholder="Type your text here..."
        value={text}
        onChange={e => setText(e.target.value)}
      />
      <button onClick={handlePredict} style={{ marginTop: 12, padding: '8px 18px', fontSize: 16 }} disabled={loading}>
        {loading ? 'Predicting...' : 'Predict'}
      </button>
      {result && (
        <div style={{ marginTop: 24 }}>
          <strong>Label:</strong> {result.label} <br />
          <strong>Score:</strong> {result.score}
        </div>
      )}
    </div>
  );
}

export default App;
