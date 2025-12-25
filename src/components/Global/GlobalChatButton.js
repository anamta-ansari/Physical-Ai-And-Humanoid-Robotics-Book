import React, { useState, useEffect } from 'react';
import RagChatBot from '../RagChatBot/RagChatBot';

const GlobalChatButton = () => {
  const [chatOpen, setChatOpen] = useState(false);

  // Only render on client side
  if (typeof window === 'undefined') {
    return null;
  }

  return (
    <div>
      <RagChatBot 
        isOpen={chatOpen} 
        onClose={() => setChatOpen(false)} 
        onToggle={() => setChatOpen(!chatOpen)} 
      />
    </div>
  );
};

export default GlobalChatButton;