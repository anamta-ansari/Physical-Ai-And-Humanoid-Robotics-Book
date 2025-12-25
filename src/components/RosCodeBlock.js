import React from 'react';
import clsx from 'clsx';
import Highlight, { defaultProps } from 'prism-react-renderer';
import { useColorMode } from '@docusaurus/theme-common';

// Component for displaying ROS code examples with proper syntax highlighting
const RosCodeBlock = ({ children, language = 'python', title, description }) => {
  const { colorMode } = useColorMode();
  
  return (
    <div className={clsx('codeBlock', 'rosCodeBlock')}>
      {title && (
        <div className="codeBlockTitle">
          <strong>{title}</strong>
        </div>
      )}
      {description && (
        <div className="codeBlockDescription">
          {description}
        </div>
      )}
      <Highlight 
        {...defaultProps} 
        code={children.trim()} 
        language={language}
        theme={colorMode === 'dark' ? undefined : undefined} // Use default theme for both modes
      >
        {({ className, style, tokens, getLineProps, getTokenProps }) => (
          <pre className={className} style={style}>
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>
                {line.map((token, key) => (
                  <span key={key} {...getTokenProps({ token })} />
                ))}
              </div>
            ))}
          </pre>
        )}
      </Highlight>
    </div>
  );
};

export default RosCodeBlock;