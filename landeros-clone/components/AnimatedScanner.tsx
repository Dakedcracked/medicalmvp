"use client";

import { useEffect, useState } from "react";

export default function AnimatedScanner() {
  const [scanProgress, setScanProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [scanType, setScanType] = useState(0);
  const [rotation, setRotation] = useState(0);

  const steps = [
    "Uploading medical image...",
    "Preprocessing DICOM data...",
    "Running neural network...",
    "Analyzing brain regions...",
    "Detecting abnormalities...",
    "Generating diagnostic report...",
    "Analysis complete!"
  ];

  const scanTypes = [
    { name: "Brain MRI", icon: "🧠", color: "from-purple-500 to-pink-500" },
    { name: "CT Scan", icon: "💀", color: "from-blue-500 to-cyan-500" },
    { name: "Chest X-Ray", icon: "🫁", color: "from-green-500 to-teal-500" }
  ];

  useEffect(() => {
    // Simulate scanning progress
    const progressInterval = setInterval(() => {
      setScanProgress((prev) => {
        if (prev >= 100) {
          return 0;
        }
        return prev + 1.5;
      });
    }, 80);

    // Update steps
    const stepInterval = setInterval(() => {
      setCurrentStep((prev) => (prev + 1) % steps.length);
    }, 2500);

    // Rotate scan type
    const typeInterval = setInterval(() => {
      setScanType((prev) => (prev + 1) % scanTypes.length);
    }, 15000);

    // 3D rotation effect
    const rotationInterval = setInterval(() => {
      setRotation((prev) => (prev + 1) % 360);
    }, 50);

    return () => {
      clearInterval(progressInterval);
      clearInterval(stepInterval);
      clearInterval(typeInterval);
      clearInterval(rotationInterval);
    };
  }, []);

  return (
    <div className="relative w-full max-w-2xl mx-auto">
      {/* Main Scanner Container */}
      <div className="bg-gradient-to-br from-gray-900 to-black rounded-3xl p-8 border border-accent/30 shadow-2xl shadow-accent/20">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-white font-semibold">Neuron AI Scanner</span>
            <span className="text-xs text-gray-400 bg-gray-800 px-2 py-1 rounded-full">
              {scanTypes[scanType].name}
            </span>
          </div>
          <div className="text-accent text-sm font-mono">{Math.floor(scanProgress)}%</div>
        </div>

        {/* Medical Image Placeholder with Scan Line */}
        <div className="relative bg-gray-800 rounded-2xl overflow-hidden mb-6 aspect-square"
          style={{
            background: `radial-gradient(circle at center, rgba(30, 30, 30, 1) 0%, rgba(0, 0, 0, 1) 100%)`
          }}
        >
          {/* Simulated Medical Scan */}
          <div className="absolute inset-0 flex items-center justify-center">
            {/* Brain/CT Scan Visualization */}
            <div 
              className="relative w-80 h-80"
              style={{
                transform: `perspective(1000px) rotateY(${rotation * 0.5}deg)`,
                transition: 'transform 0.05s linear'
              }}
            >
              {/* Brain Outline */}
              <div className="absolute inset-0 flex items-center justify-center">
                {/* Outer skull */}
                <div className="absolute w-64 h-72 border-2 border-gray-600/40 rounded-full"></div>
                
                {/* Brain hemispheres */}
                <div className="absolute w-56 h-64 bg-gradient-to-br from-gray-700/30 to-gray-600/20 rounded-full blur-sm"></div>
                
                {/* Left hemisphere */}
                <div className="absolute left-1/4 top-1/4 w-24 h-48 bg-gray-600/30 rounded-l-full blur-md"></div>
                
                {/* Right hemisphere */}
                <div className="absolute right-1/4 top-1/4 w-24 h-48 bg-gray-600/30 rounded-r-full blur-md"></div>
                
                {/* Brain folds/sulci */}
                {[...Array(8)].map((_, i) => (
                  <div
                    key={i}
                    className="absolute w-1 bg-gray-500/20 blur-sm"
                    style={{
                      height: `${40 + i * 5}px`,
                      left: `${30 + i * 10}%`,
                      top: `${25 + (i % 3) * 15}%`,
                      transform: `rotate(${i * 15}deg)`,
                    }}
                  />
                ))}

                {/* CT scan slices effect */}
                {scanProgress > 20 && [...Array(5)].map((_, i) => (
                  <div
                    key={i}
                    className="absolute left-0 right-0 h-px bg-cyan-400/30"
                    style={{
                      top: `${20 + i * 15}%`,
                      opacity: scanProgress > (20 + i * 15) ? 0.6 : 0,
                      transition: 'opacity 0.3s'
                    }}
                  />
                ))}

                {/* Ventricles (brain cavities) */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-12 h-16 bg-gray-800/40 rounded-full blur-sm"></div>
                
                {/* Cerebellum */}
                <div className="absolute bottom-1/4 left-1/2 -translate-x-1/2 w-20 h-12 bg-gray-700/30 rounded-full blur-sm"></div>
              </div>

              {/* Rotating scan ring */}
              <div 
                className="absolute inset-0 border-2 border-accent/30 rounded-full"
                style={{
                  transform: `rotate(${rotation}deg)`,
                  boxShadow: '0 0 30px rgba(59, 130, 246, 0.3)'
                }}
              >
                <div className="absolute top-0 left-1/2 w-2 h-2 bg-accent rounded-full -translate-x-1/2 -translate-y-1/2"></div>
              </div>
            </div>
          </div>

          {/* Animated Scan Line */}
          <div 
            className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-accent to-transparent shadow-lg shadow-accent/50 transition-all duration-100"
            style={{ 
              top: `${scanProgress}%`,
              boxShadow: '0 0 20px rgba(59, 130, 246, 0.8)'
            }}
          >
            <div className="absolute inset-0 bg-accent blur-md"></div>
          </div>

          {/* AI Detection Markers */}
          {scanProgress > 25 && (
            <div className="absolute top-1/4 left-1/3 w-20 h-20 border-2 border-yellow-500/60 rounded-full animate-pulse"
              style={{
                boxShadow: '0 0 20px rgba(234, 179, 8, 0.4)',
                animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite'
              }}
            >
              <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-yellow-500 text-black text-xs px-3 py-1 rounded-full whitespace-nowrap font-semibold shadow-lg">
                Analyzing Region
              </div>
            </div>
          )}

          {scanProgress > 50 && (
            <div className="absolute top-1/2 right-1/3 w-16 h-16 border-2 border-cyan-400/60 rounded-full animate-pulse"
              style={{
                boxShadow: '0 0 20px rgba(34, 211, 238, 0.4)'
              }}
            >
              <div className="absolute -top-8 right-0 bg-cyan-400 text-black text-xs px-3 py-1 rounded-full whitespace-nowrap font-semibold shadow-lg">
                Processing
              </div>
            </div>
          )}

          {scanProgress > 75 && (
            <div className="absolute bottom-1/3 left-1/2 w-14 h-14 border-2 border-green-500/60 rounded-full animate-pulse"
              style={{
                boxShadow: '0 0 20px rgba(34, 197, 94, 0.4)'
              }}
            >
              <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 bg-green-500 text-white text-xs px-3 py-1 rounded-full whitespace-nowrap font-semibold shadow-lg">
                ✓ Normal
              </div>
            </div>
          )}

          {/* Grid Overlay */}
          <div className="absolute inset-0 opacity-10">
            <div className="grid grid-cols-8 grid-rows-8 h-full">
              {[...Array(64)].map((_, i) => (
                <div key={i} className="border border-accent/20"></div>
              ))}
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-accent to-blue-600 transition-all duration-100 rounded-full"
              style={{ width: `${scanProgress}%` }}
            >
              <div className="h-full w-full bg-white/20 animate-pulse"></div>
            </div>
          </div>
        </div>

        {/* Status Text */}
        <div className="space-y-3">
          <div className="flex items-center gap-3 text-sm">
            <div className="w-2 h-2 bg-accent rounded-full animate-pulse"></div>
            <span className="text-gray-300 font-mono">{steps[currentStep]}</span>
          </div>

          {/* AI Processing Indicators */}
          <div className="grid grid-cols-3 gap-3 mt-4">
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg p-3 border border-gray-700 hover:border-accent/50 transition-all group">
              <div className="flex items-center justify-between mb-1">
                <div className="text-xs text-gray-400">Confidence</div>
                <div className="w-2 h-2 bg-accent rounded-full animate-pulse"></div>
              </div>
              <div className="text-2xl font-bold bg-gradient-to-r from-accent to-blue-400 bg-clip-text text-transparent">
                {(85 + scanProgress * 0.1).toFixed(1)}%
              </div>
              <div className="h-1 bg-gray-700 rounded-full mt-2 overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-accent to-blue-400 transition-all duration-300"
                  style={{ width: `${85 + scanProgress * 0.1}%` }}
                ></div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg p-3 border border-gray-700 hover:border-green-500/50 transition-all group">
              <div className="flex items-center justify-between mb-1">
                <div className="text-xs text-gray-400">Processing</div>
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              </div>
              <div className="text-2xl font-bold text-green-500">
                {(scanProgress / 50).toFixed(1)}s
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {scanProgress < 100 ? 'In Progress...' : 'Complete'}
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg p-3 border border-gray-700 hover:border-purple-500/50 transition-all group">
              <div className="flex items-center justify-between mb-1">
                <div className="text-xs text-gray-400">Regions</div>
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></div>
              </div>
              <div className="text-2xl font-bold text-purple-500">
                {Math.min(Math.floor(scanProgress / 12.5), 8)}/8
              </div>
              <div className="flex gap-1 mt-2">
                {[...Array(8)].map((_, i) => (
                  <div 
                    key={i}
                    className={`h-1 flex-1 rounded-full transition-all duration-300 ${
                      i < Math.floor(scanProgress / 12.5) 
                        ? 'bg-purple-500' 
                        : 'bg-gray-700'
                    }`}
                  ></div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Status */}
        <div className="mt-6 pt-6 border-t border-gray-800 flex items-center justify-between text-xs text-gray-500">
          <span>DenseNet121 Neural Network</span>
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 bg-green-500 rounded-full"></span>
            GPU Accelerated
          </span>
        </div>
      </div>

      {/* Floating Particles */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-0 w-2 h-2 bg-accent rounded-full animate-ping opacity-20"></div>
        <div className="absolute top-1/2 right-0 w-2 h-2 bg-blue-500 rounded-full animate-ping opacity-20" style={{ animationDelay: '0.5s' }}></div>
        <div className="absolute bottom-1/4 left-1/3 w-2 h-2 bg-accent rounded-full animate-ping opacity-20" style={{ animationDelay: '1s' }}></div>
      </div>
    </div>
  );
}
