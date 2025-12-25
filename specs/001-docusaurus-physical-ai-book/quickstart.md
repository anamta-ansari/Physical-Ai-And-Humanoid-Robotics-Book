# Quickstart Guide: Physical AI & Humanoid Robotics Book

## Overview
This guide will help you get the Physical AI & Humanoid Robotics book up and running on your local development environment.

## Prerequisites
- Node.js version 18.x or higher
- npm or yarn package manager
- Git for version control

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Dependencies
```bash
npm install
# or
yarn install
```

### 3. Start the Development Server
```bash
npm run start
# or
yarn start
```

This command starts a local development server and opens the book in your browser. Most changes are reflected live without having to restart the server.

## Project Structure
```
/
├── docs/                # All book content organized by parts and chapters
│   ├── part1/
│   │   ├── chapter1.md
│   │   ├── chapter2.md
│   │   └── chapter3.md
│   ├── part2/
│   │   ├── chapter4.md
│   │   ├── chapter5.md
│   │   └── chapter6.md
│   └── ...              # Additional parts and chapters
├── src/
│   ├── components/      # Custom React components for book
│   ├── pages/           # Custom pages if needed
│   └── css/
│       └── custom.css   # Custom styles including animations
├── static/
│   └── img/             # All book images and diagrams
├── docusaurus.config.js # Docusaurus configuration
├── sidebars.js          # Sidebar navigation configuration
└── package.json         # Project dependencies
```

## Adding New Content

### Creating a New Chapter
1. Create a new markdown file in the appropriate part directory (e.g., `docs/part1/new-chapter.md`)
2. Add the new chapter to the `sidebars.js` file to make it appear in the navigation

### Chapter Frontmatter Template
```markdown
---
title: Chapter Title
sidebar_position: X
description: Brief description of the chapter content
---

# Chapter Title

## Section Header

Content goes here...
```

### Custom Styling
- Add custom CSS to `src/css/custom.css`
- This is where the hero section animations and other custom styles are defined

## Configuration

### Docusaurus Configuration
The `docusaurus.config.js` file contains:
- Site metadata (title, tagline, favicon)
- Theme configuration
- Plugin settings
- Navigation configuration

### Sidebar Configuration
The `sidebars.js` file defines the navigation structure for the book, organizing content by parts and chapters.

## Building for Production

To build the book for deployment:

```bash
npm run build
# or
yarn build
```

This command generates static content in the `build` directory, which can be served using any static hosting service.

## Deployment

The book can be deployed to various platforms:
- GitHub Pages
- Netlify
- Vercel
- Any static hosting service

For GitHub Pages deployment, use:
```bash
npm run deploy
# or
yarn deploy
```

## Troubleshooting

### Common Issues

1. **Page not loading after changes**:
   - Restart the development server with `npm run start`

2. **Images not showing**:
   - Ensure images are placed in the `static/img/` directory
   - Use the correct path format in markdown: `![alt text](/img/image-name.jpg)`

3. **Navigation not working**:
   - Check that the page is properly listed in `sidebars.js`
   - Verify the `sidebar_position` value is correctly set

## Contributing

To contribute to the book:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-chapter`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing chapter on topic'`)
5. Push to the branch (`git push origin feature/amazing-chapter`)
6. Open a Pull Request