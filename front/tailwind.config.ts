import type { Config } from 'tailwindcss';

const config = {
  darkMode: ['class'],
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  prefix: '',
  theme: {
    container: {
      center: true,
      padding: '2rem',
      screens: {
        '2xl': '1400px',
      },
    },
    extend: {
      colors: {
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          '50': '#e6f2ff',
          '100': '#cce4ff',
          '200': '#99c9ff',
          '300': '#66adff',
          '400': '#3392ff',
          '500': '#0084ff',
          '600': '#0076e6',
          '700': '#0068cc',
          '800': '#005bb3',
          '900': '#004d99',
        },
        secondary: {
          '50': '#e5fbff',
          '100': '#c0f5ff',
          '200': '#8aefff',
          '300': '#54e9ff',
          '400': '#1ee3ff',
          '500': '#00c8ff',
          '600': '#00b4e6',
          '700': '#009fcc',
          '800': '#008ab3',
          '900': '#007699',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          '50': '#ecebff',
          '100': '#d2d0ff',
          '200': '#a5a0ff',
          '300': '#7871ff',
          '400': '#4b41ff',
          '500': '#3334ff',
          '600': '#2d2ee6',
          '700': '#2728cc',
          '800': '#2121b3',
          '900': '#1b1b99',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        neutral: {
          '50': '#fafbfc',
          '100': '#f2f4f6',
          '200': '#e5e8ec',
          '300': '#cfd4da',
          '400': '#b9c0c8',
          '500': '#a3acb6',
          '600': '#8e98a3',
          '700': '#78828e',
          '800': '#626c79',
          '900': '#4c5664',
        },
        success: '#12b76a',
        warning: '#f79009',
        error: '#f04438',
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' },
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":
          "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
    },
  },
  plugins: [require('tailwindcss-animate'), require('@tailwindcss/typography')],
} satisfies Config;

export default config;
