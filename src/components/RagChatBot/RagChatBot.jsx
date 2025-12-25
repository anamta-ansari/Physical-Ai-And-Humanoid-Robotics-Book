import React, { useState, useEffect, useRef } from 'react';
import './styles.css';

const RagChatBot = ({ isOpen, onClose, onToggle }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const messagesEndRef = useRef(null);

  // Function to get selected text from the page
  const getSelectedText = () => {
    const text = window.getSelection().toString().trim();
    if (text) {
      setSelectedText(text);
      // Auto-focus the input and add the selected text
      setTimeout(() => {
        setInputValue(prev => prev ? `${prev} ${text}` : text);
      }, 100);
    }
  };

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle sending a message
  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = { id: Date.now(), text: inputValue, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    
    const currentSelectedText = selectedText;
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputValue,
          selected_text: currentSelectedText || null,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      
      // Process the streaming response if needed
      const botMessage = {
        id: Date.now() + 1,
        text: data.response,
        sender: 'bot',
        sources: data.sources || [],
      };
      
      setMessages(prev => [...prev, botMessage]);
      // Clear selected text after sending
      setSelectedText('');
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle key press (Enter to send)
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Clear chat
  const handleClear = () => {
    setMessages([]);
  };

  if (!isOpen) {
    return (
      <button className="chat-float-button" onClick={onToggle}>
        💬
      </button>
    );
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h3>Book Assistant</h3>
        <div className="chat-controls">
          <button onClick={getSelectedText} className="select-text-btn" title="Use selected text">
            📝
          </button>
          <button onClick={handleClear} className="clear-btn" title="Clear chat">
            🗑️
          </button>
          <button onClick={onClose} className="close-btn" title="Close">
            ✕
          </button>
        </div>
      </div>
      
      {selectedText && (
        <div className="selected-text-preview">
          <strong>Using selected text:</strong> "{selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}"
          <button onClick={() => setSelectedText('')} className="remove-selection">Remove</button>
        </div>
      )}
      
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <p>Hello! I'm your book assistant. Ask me anything about the Physical AI & Humanoid Robotics book.</p>
            <p>You can also select text on the page and click the button to ask about it specifically!</p>
          </div>
        ) : (
          messages.map((message) => (
            <div key={message.id} className={`message ${message.sender}`}>
              <div className="message-text">{message.text}</div>
              {message.sources && message.sources.length > 0 && (
                <div className="message-sources">
                  Sources: {message.sources.join(', ')}
                </div>
              )}
            </div>
          ))
        )}
        {isLoading && (
          <div className="message bot">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="chat-input-area">
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about the book content..."
          className="chat-input"
          rows="2"
        />
        <button 
          onClick={handleSend} 
          disabled={isLoading || !inputValue.trim()}
          className="send-button"
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
};

export default RagChatBot;