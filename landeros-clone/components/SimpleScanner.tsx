"use client";

import { useEffect, useState } from "react";

export default function SimpleScanner() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          setTimeout(() => setProgress(0), 1000);
          return 100;
        }
        return prev + 1.5;
      });
    }, 50);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="max-w-5xl mx-auto">
      <div className="grid md:grid-cols-2 gap-12 items-center">
        {/* Left: Cute Baby Skeleton Scan */}
        <div className="relative group cursor-pointer">
          <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-cyan-400 rounded-3xl blur opacity-25 group-hover:opacity-50 transition duration-1000"></div>
          <div className="relative bg-gray-900 rounded-3xl p-1 shadow-2xl">
            {/* Scan Window */}
            <div className="relative bg-black rounded-[20px] overflow-hidden aspect-[3/4]">
              
              {/* Grid Background */}
              <div className="absolute inset-0 opacity-20" 
                style={{
                  backgroundImage: 'linear-gradient(#333 1px, transparent 1px), linear-gradient(90deg, #333 1px, transparent 1px)',
                  backgroundSize: '20px 20px'
                }}>
              </div>

              {/* Baby Skeleton SVG */}
              <div className="absolute inset-0 flex items-center justify-center">
                <svg viewBox="0 0 200 300" className="w-3/4 h-3/4 drop-shadow-[0_0_15px_rgba(255,255,255,0.3)]">
                  {/* Skull */}
                  <path fill="white" d="M100 40 c-25 0 -45 20 -45 45 c0 25 15 40 35 45 v10 h20 v-10 c20 -5 35 -20 35 -45 c0 -25 -20 -45 -45 -45 z M85 70 a5 5 0 0 1 10 0 a5 5 0 0 1 -10 0 M105 70 a5 5 0 0 1 10 0 a5 5 0 0 1 -10 0 M95 90 c2 2 8 2 10 0" opacity="0.9"/>
                  {/* Ribcage */}
                  <path fill="none" stroke="white" strokeWidth="8" strokeLinecap="round" d="M100 100 v60 M70 110 q30 -10 60 0 M75 125 q25 -10 50 0 M80 140 q20 -10 40 0 M85 155 q15 -10 30 0" opacity="0.8"/>
                  {/* Hips */}
                  <path fill="white" d="M80 170 q20 10 40 0 l5 20 q-25 10 -50 0 z" opacity="0.9"/>
                  {/* Legs */}
                  <path fill="none" stroke="white" strokeWidth="8" strokeLinecap="round" d="M85 190 l-10 40 l-5 30 M115 190 l10 40 l5 30" opacity="0.8"/>
                  {/* Arms (Waving) */}
                  <path fill="none" stroke="white" strokeWidth="8" strokeLinecap="round" d="M70 110 l-20 20 l-10 30 M130 110 l20 -20 l10 -30" opacity="0.8"/>
                </svg>
              </div>

              {/* Scanning Beam */}
              <div 
                className="absolute left-0 right-0 h-1 bg-cyan-400 shadow-[0_0_30px_rgba(34,211,238,1)] transition-all duration-75 z-10"
                style={{ top: `${progress}%` }}
              >
                <div className="absolute right-0 -top-2 bg-cyan-500 text-[10px] text-black font-bold px-1 rounded">
                  SCANNING
                </div>
              </div>

              {/* Floating Particles */}
              <div className="absolute inset-0 pointer-events-none">
                <div className="absolute top-10 left-10 w-1 h-1 bg-cyan-400 rounded-full animate-ping"></div>
                <div className="absolute top-40 right-20 w-1 h-1 bg-cyan-400 rounded-full animate-ping delay-700"></div>
                <div className="absolute bottom-20 left-1/2 w-1 h-1 bg-cyan-400 rounded-full animate-ping delay-300"></div>
              </div>

              {/* Results Overlay */}
              {progress > 90 && (
                <div className="absolute inset-0 bg-black/60 flex items-center justify-center animate-in fade-in duration-300">
                  <div className="bg-white/10 backdrop-blur-md border border-white/20 p-6 rounded-2xl text-center">
                    <div className="text-4xl mb-2">✨</div>
                    <div className="text-2xl font-bold text-white mb-1">Healthy!</div>
                    <div className="text-cyan-300 text-sm">Confidence: 99.8%</div>
                  </div>
                </div>
              )}
            </div>

            {/* Status Bar */}
            <div className="mt-4 px-2 flex justify-between items-center font-mono text-xs text-gray-400">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${progress < 100 ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
                SYSTEM ACTIVE
              </div>
              <div>ID: BABY_SCAN_01</div>
            </div>
          </div>
        </div>

        {/* Right: Content */}
        <div className="space-y-8">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-100 text-blue-600 text-sm font-semibold mb-4">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
              </span>
              Watch AI in Action
            </div>
            <h3 className="text-4xl font-bold mb-4 text-gray-900">
              Pediatric-Friendly <br/>
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-cyan-500">
                Intelligent Scanning
              </span>
            </h3>
            <p className="text-lg text-gray-600 leading-relaxed">
              Our advanced AI models are trained to be gentle yet precise. See how our system analyzes pediatric scans in real-time, identifying key anatomical markers with near-perfect accuracy.
            </p>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 gap-6">
            <div className="p-4 rounded-2xl bg-blue-50 border border-blue-100">
              <div className="text-3xl font-bold text-blue-600 mb-1">98.5%</div>
              <div className="text-sm text-gray-600 font-medium">Model Accuracy</div>
            </div>
            <div className="p-4 rounded-2xl bg-cyan-50 border border-cyan-100">
              <div className="text-3xl font-bold text-cyan-600 mb-1">0.2s</div>
              <div className="text-sm text-gray-600 font-medium">Processing Speed</div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4">
            <a 
              href="/analyze" 
              className="flex-1 bg-gray-900 text-white text-center px-8 py-4 rounded-xl font-semibold hover:bg-gray-800 transition-all hover:-translate-y-1 shadow-lg shadow-gray-900/20"
            >
              Try Live Demo
            </a>
            <button className="flex-1 bg-white text-gray-900 border border-gray-200 px-8 py-4 rounded-xl font-semibold hover:bg-gray-50 transition-all hover:-translate-y-1">
              View Report
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
