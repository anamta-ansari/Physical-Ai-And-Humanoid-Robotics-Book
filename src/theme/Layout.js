import React from 'react';
import Layout from '@theme-original/Layout';
import GlobalChatButton from '@site/src/components/Global/GlobalChatButton';

export default function LayoutWrapper(props) {
  return (
    <>
      <Layout {...props}>
        {props.children}
        <GlobalChatButton />
      </Layout>
    </>
  );
}