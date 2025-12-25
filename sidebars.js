// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'part1/chapter1',
    'part1/chapter2',
    'part1/chapter3',
    {
      type: 'category',
      label: 'Robotic Nervous System (ROS 2)',
      items: [
        'part2/chapter4',
        'part2/chapter5',
        'part2/chapter6',
      ],
    },
    {
      type: 'category',
      label: 'Digital Twin & Simulation',
      items: [
        'part3/chapter7',
        'part3/chapter8',
      ],
    },
    {
      type: 'category',
      label: 'AI-Robot Brain (NVIDIA ISAAC)',
      items: [
        'part4/chapter9',
        'part4/chapter10',
        'part4/chapter11',
      ],
    },
    {
      type: 'category',
      label: 'Humanoid Robot Engineering',
      items: [
        'part5/chapter12',
        'part5/chapter13',
        'part5/chapter14',
      ],
    },
    {
      type: 'category',
      label: 'Vision-Language-Action Robotics',
      items: [
        'part6/chapter15',
        'part6/chapter16',
        'part6/chapter17',
        'part6/chapter18',
      ],
    },
    {
      type: 'category',
      label: 'Hardware Requirements & Lab Setup',
      items: [
        'part7/chapter19',
        'part7/chapter20',
        'part7/chapter21',
      ],
    },
    {
      type: 'category',
      label: 'Implementation, Assessments & Projects',
      items: [
        'part8/chapter23',
        'part8/chapter24',
      ],
    },
  ],
};

export default sidebars;