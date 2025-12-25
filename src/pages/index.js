import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className={styles.heroOverlay}></div>
      <div className={clsx('container', styles.heroContent)}>
        <h1 className="hero__title">Physical AI & Humanoid Robotics</h1>
        <p className="hero__subtitle">
          A complete handbook on embodied intelligence, humanoid robot systems, mechanisms, perception, control, and real-world robotics engineering.
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/part1/chapter1">
            Read the Book
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/resources">
            Download Resources
          </Link>
        </div>
      </div>
    </header>
  );
}

// Chapter Overview Card Component
function ChapterCard({ title, icon }) {
  return (
    <div className={styles.chapterCard}>
      <div className={styles.cardIcon}>{icon}</div>
      <h3>{title}</h3>
    </div>
  );
}

// Tutorial Card Component
function TutorialCard({ title, icon }) {
  return (
    <div className={styles.tutorialCard}>
      <div className={styles.cardIcon}>{icon}</div>
      <h3>{title}</h3>
    </div>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics`}
      description="A complete handbook on embodied intelligence, humanoid robot systems, mechanisms, perception, control, and real-world robotics engineering.">
      <HomepageHeader />
      <main>
        {/* Chapter Overview Grid */}
        <section className={styles.chapterOverview}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Book Contents</h2>
            <div className={styles.chapterGrid}>
              <ChapterCard title="Introduction to Physical AI" icon="🤖" />
              <ChapterCard title="Humanoid Robot Architecture" icon="🏗️" />
              <ChapterCard title="Sensors & Perception" icon="👁️" />
              <ChapterCard title="Motion Planning" icon="🧭" />
              <ChapterCard title="Control Systems" icon="⚙️" />
              <ChapterCard title="Robot Learning" icon="🧠" />
              <ChapterCard title="Actuation & Hardware" icon="⚡" />
              <ChapterCard title="Advanced Topics" icon="🔬" />
            </div>
          </div>
        </section>

        {/* About the Book Section */}
        <section className={styles.aboutSection}>
          <div className="container">
            <div className={styles.aboutContent}>
              <div className={styles.aboutText}>
                <h2>About the Book</h2>
                <p>
                  This comprehensive guide covers all aspects of Physical AI and Humanoid Robotics,
                  from foundational concepts to advanced implementations. It provides in-depth
                  coverage of embodied AI foundations, humanoid robot kinematics, and real-world
                  robotics engineering principles.
                </p>
                <p>
                  Designed for students, researchers, and professionals, this book offers practical
                  insights into developing and deploying humanoid robots in various applications.
                </p>
              </div>
              <div className={styles.aboutImage}>
                <img
                  src="https://images.stockcake.com/public/7/4/1/741421eb-26b4-4210-aab8-21cfba2bb8cd/neon-robot-profile-stockcake.jpg"
                  alt="Futuristic holographic blueprint of a humanoid robot with neon glow"
                  className={styles.robotBlueprint}
                />
              </div>
            </div>
          </div>
        </section>

        {/* Tutorials & Labs Section */}
        <section className={styles.tutorialsSection}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Tutorials & Labs</h2>
            <div className={styles.tutorialGrid}>
              <TutorialCard title="Build a Legged Robot" icon="🦾" />
              <TutorialCard title="Design a Balance Controller" icon="⚖️" />
              <TutorialCard title="Robot Vision Lab" icon="👁️" />
              <TutorialCard title="ROS 2 Integration" icon="🔗" />
            </div>
          </div>
        </section>

        {/* Featured Visuals/Robotics Gallery Section */}
        <section className={styles.featuredStrip}>
          <div className="container">
            <h2 className={styles.sectionTitle}>Featured Visuals</h2>
            <div className={styles.featuredImages}>
              <img
                src="https://thumbs.dreamstime.com/b/sleek-futuristic-robot-high-tech-lab-under-neon-lighting-photographed-wide-angle-lens-appears-to-be-experiencing-396923248.jpg"
                alt="Real humanoid in lab with neon lighting"
                className={styles.featuredImage}
              />
              <img
                src="https://thumbs.dreamstime.com/b/digital-circuit-board-microprocessor-motherboard-computer-hardware-neon-purple-blue-glowing-light-futuristic-technology-297149396.jpg"
                alt="Neon AI circuit closeup with purple/blue glow"
                className={styles.featuredImage}
              />
              <img
                src="https://thumbs.dreamstime.com/b/futuristic-humanoid-robot-sleek-design-stands-glowing-high-tech-environment-filled-digital-circuits-neon-lights-370945883.jpg"
                alt="Robot/AI visual in high-tech environment"
                className={styles.featuredImage}
              />
              <img
                src="https://thumbs.dreamstime.com/b/abstract-digital-circuit-board-neon-lights-bokeh-vibrant-abstract-visualization-digital-circuit-board-glowing-405629778.jpg"
                alt="Abstract digital circuit board with neon lights"
                className={styles.featuredImage}
              />
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}