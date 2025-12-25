import React, { useState } from 'react';
import Layout from '@theme/Layout';
import RagChatBot from '../components/RagChatBot/RagChatBot';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

export default function ChatPage() {
  const { siteConfig } = useDocusaurusContext();
  const [chatOpen, setChatOpen] = useState(true);
  
  return (
    <Layout
      title={`Chat - ${siteConfig.title}`}
      description="Chat with the book assistant">
      <main style={{ 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center', 
        justifyContent: 'center',
        minHeight: '80vh',
        padding: '20px'
      }}>
        <div style={{ 
          width: '100%', 
          maxWidth: '800px', 
          margin: '0 auto',
          textAlign: 'center'
        }}>
          <h1>Book Assistant Chat</h1>
          <p>Ask questions about the Physical AI & Humanoid Robotics book content</p>
          
          {/* Render the chatbot component in full-page mode */}
          <div style={{ marginTop: '20px', height: '70vh', width: '100%' }}>
            <RagChatBot 
              isOpen={true} 
              onClose={() => {}} 
              onToggle={() => setChatOpen(!chatOpen)} 
            />
          </div>
        </div>
      </main>
    </Layout>
  );
}