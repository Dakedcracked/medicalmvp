"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

export default function NotFound() {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [glitchActive, setGlitchActive] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener("mousemove", handleMouseMove);

    // Random glitch effect
    const glitchInterval = setInterval(() => {
      setGlitchActive(true);
      setTimeout(() => setGlitchActive(false), 200);
    }, 3000);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      clearInterval(glitchInterval);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 flex items-center justify-center relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 opacity-20">
        <div 
          className="absolute w-96 h-96 bg-accent/30 rounded-full blur-3xl animate-pulse"
          style={{
            left: `${mousePosition.x - 192}px`,
            top: `${mousePosition.y - 192}px`,
            transition: 'all 0.3s ease-out'
          }}
        ></div>
        <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute bottom-1/4 left-1/4 w-96 h-96 bg-pink-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      {/* Grid Pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="grid grid-cols-12 grid-rows-12 h-full">
          {[...Array(144)].map((_, i) => (
            <div key={i} className="border border-accent/20"></div>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="relative z-10 text-center px-6 max-w-4xl">
        {/* 404 Number */}
        <div className="mb-8 relative">
          <h1 
            className={`text-[200px] md:text-[300px] font-bold leading-none ${
              glitchActive ? 'glitch' : ''
            }`}
            style={{
              background: 'linear-gradient(135deg, #3B82F6 0%, #8B5CF6 50%, #EC4899 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              textShadow: glitchActive ? '0 0 50px rgba(59, 130, 246, 0.5)' : 'none'
            }}
          >
            404
          </h1>
          
          {/* Floating Particles */}
          <div className="absolute inset-0 pointer-events-none">
            {[...Array(20)].map((_, i) => (
              <div
                key={i}
                className="absolute w-2 h-2 bg-accent rounded-full animate-ping"
                style={{
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                  animationDelay: `${Math.random() * 2}s`,
                  animationDuration: `${2 + Math.random() * 2}s`,
                  opacity: 0.3
                }}
              ></div>
            ))}
          </div>
        </div>

        {/* Error Message */}
        <div className="mb-8 space-y-4">
          <h2 className="text-4xl md:text-6xl font-bold text-white mb-4">
            Page Not Found
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto leading-relaxed">
            Oops! Looks like this page got lost in the neural network. 
            The AI couldn't locate what you're looking for.
          </p>
        </div>

        {/* Animated Brain Icon */}
        <div className="mb-12 flex justify-center">
          <div className="relative">
            <div className="text-9xl animate-bounce">🧠</div>
            <div className="absolute inset-0 bg-accent/20 rounded-full blur-3xl animate-pulse"></div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
          <Link
            href="/"
            className="group relative px-8 py-4 bg-gradient-to-r from-accent to-blue-600 text-white rounded-full font-semibold text-lg overflow-hidden transition-all hover:scale-105 hover:shadow-2xl hover:shadow-accent/50"
          >
            <span className="relative z-10 flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
              </svg>
              Go Home
            </span>
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 opacity-0 group-hover:opacity-100 transition-opacity"></div>
          </Link>

          <Link
            href="/contact"
            className="px-8 py-4 bg-white/10 backdrop-blur-sm text-white border-2 border-white/20 rounded-full font-semibold text-lg hover:bg-white/20 hover:border-accent transition-all hover:scale-105"
          >
            Contact Support
          </Link>
        </div>

        {/* Quick Links */}
        <div className="mt-16 pt-8 border-t border-gray-800">
          <p className="text-gray-500 mb-4">Quick Links</p>
          <div className="flex flex-wrap gap-4 justify-center">
            <Link href="/solutions" className="text-gray-400 hover:text-accent transition-colors">
              Solutions
            </Link>
            <span className="text-gray-700">•</span>
            <Link href="/analyze" className="text-gray-400 hover:text-accent transition-colors">
              Analyze
            </Link>
            <span className="text-gray-700">•</span>
            <Link href="/blog" className="text-gray-400 hover:text-accent transition-colors">
              Blog
            </Link>
            <span className="text-gray-700">•</span>
            <Link href="/pricing" className="text-gray-400 hover:text-accent transition-colors">
              Pricing
            </Link>
            <span className="text-gray-700">•</span>
            <Link href="/dashboard" className="text-gray-400 hover:text-accent transition-colors">
              Dashboard
            </Link>
          </div>
        </div>

        {/* Error Code */}
        <div className="mt-8 text-xs text-gray-600 font-mono">
          ERROR_CODE: NEURON_404_PAGE_NOT_FOUND
        </div>
      </div>

      {/* Floating Elements */}
      <div className="absolute top-10 left-10 text-6xl opacity-20 animate-spin-slow">⚕️</div>
      <div className="absolute bottom-10 right-10 text-6xl opacity-20 animate-spin-slow" style={{ animationDirection: 'reverse' }}>🔬</div>
      <div className="absolute top-1/2 right-20 text-4xl opacity-20 animate-bounce" style={{ animationDelay: '0.5s' }}>💊</div>
      <div className="absolute bottom-1/4 left-20 text-4xl opacity-20 animate-bounce" style={{ animationDelay: '1s' }}>🩺</div>

      <style jsx>{`
        @keyframes spin-slow {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }
        .animate-spin-slow {
          animation: spin-slow 20s linear infinite;
        }
        .glitch {
          animation: glitch 0.3s ease-in-out;
        }
        @keyframes glitch {
          0% {
            transform: translate(0);
          }
          20% {
            transform: translate(-5px, 5px);
          }
          40% {
            transform: translate(-5px, -5px);
          }
          60% {
            transform: translate(5px, 5px);
          }
          80% {
            transform: translate(5px, -5px);
          }
          100% {
            transform: translate(0);
          }
        }
      `}</style>
    </div>
  );
}
