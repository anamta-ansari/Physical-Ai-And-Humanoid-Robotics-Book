// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Mastering Embodied Intelligence and Humanoid Systems',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://your-docusaurus-site.example.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'speckit', // Usually your GitHub org/user name.
  projectName: 'physical-ai-humanoid-robotics-book', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/speckit/physical-ai-humanoid-robotics-book/edit/main/',
          remarkPlugins: [require('remark-math')],
          rehypePlugins: [require('rehype-katex')],
        },
        blog: false, // Disable blog if not needed
        theme: {
          customCss: './src/css/custom.css',
        },
        gtag: {
          trackingID: 'G-XXXXXXXXXX',
          anonymizeIP: true,
        },
      }),
    ],
  ],

  themes: [
    // Add search functionality (using built-in search instead of Algolia)
    // [
    //   '@docusaurus/theme-search-algolia',
    //   {
    //     // The application ID provided by Algolia
    //     appId: 'YOUR_APP_ID',
    //     // Public API key: it is safe to commit it
    //     apiKey: 'YOUR_SEARCH_API_KEY',
    //     indexName: 'physical-ai-humanoid-robotics-book',
    //     contextualSearch: true,
    //     searchPagePath: 'search',
    //   },
    // ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeData} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        hideOnScroll: false, // Ensure navbar doesn't hide on scroll
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Book',
          },
          {
            href: 'https://github.com/speckit/physical-ai-humanoid-robotics-book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Chapters',
                to: '/docs/part1/chapter1',
              },
              {
                label: 'Tutorials',
                to: '/tutorials',
              },
              {
                label: 'Labs',
                to: '/labs',
              },
            ],
          },
          {
            title: 'Resources',
            items: [
              {
                label: 'License',
                to: '/license',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/speckit/physical-ai-humanoid-robotics-book',
              },
              {
                label: 'Contact',
                href: 'mailto:contact@physicalairobotics.com',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Blog',
                to: '/blog',
              },
              {
                label: 'Resources',
                to: '/resources',
              },
            ],
          },
        ],
        copyright: `Made with Docusaurus. Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Book.`,
      },
      docs: {
        sidebar: {
          hideable: true,
          autoCollapseCategories: false,
        },
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;