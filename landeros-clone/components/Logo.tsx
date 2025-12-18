import React from "react";

export default function Logo({ className = "w-10 h-10" }: { className?: string }) {
  return (
    <div className={`relative flex items-center justify-center ${className}`}>
      <svg
        viewBox="0 0 100 100"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full overflow-visible"
      >
        <defs>
          <linearGradient id="neuronGradient" x1="0" y1="0" x2="100" y2="100" gradientUnits="userSpaceOnUse">
            <stop stopColor="#2DD4BF" />
            <stop offset="1" stopColor="#0D9488" />
          </linearGradient>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Outer Ring (Rotating) */}
        <g className="animate-[spin_10s_linear_infinite] origin-center opacity-30">
          <circle cx="50" cy="50" r="45" stroke="url(#neuronGradient)" strokeWidth="1" strokeDasharray="10 10" />
        </g>

        {/* Inner Neural Network (Futuristic N) */}
        <g filter="url(#glow)">
          {/* Left Vertical */}
          <path
            d="M30 20 V80"
            stroke="url(#neuronGradient)"
            strokeWidth="8"
            strokeLinecap="round"
            className="animate-[pulse_3s_ease-in-out_infinite]"
          />
          {/* Diagonal Connection */}
          <path
            d="M30 25 L70 75"
            stroke="url(#neuronGradient)"
            strokeWidth="8"
            strokeLinecap="round"
            className="opacity-80"
          />
          {/* Right Vertical */}
          <path
            d="M70 20 V80"
            stroke="url(#neuronGradient)"
            strokeWidth="8"
            strokeLinecap="round"
            className="animate-[pulse_3s_ease-in-out_infinite] delay-500"
          />
          
          {/* Nodes (Dots) */}
          <circle cx="30" cy="25" r="6" fill="white" />
          <circle cx="70" cy="75" r="6" fill="white" />
          <circle cx="30" cy="80" r="4" fill="#0D9488" />
          <circle cx="70" cy="20" r="4" fill="#0D9488" />
        </g>

        {/* Data Particles (Floating) */}
        <circle cx="50" cy="50" r="2" fill="#2DD4BF" className="animate-ping" />
      </svg>
    </div>
  );
}
