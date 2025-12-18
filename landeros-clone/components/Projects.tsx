"use client";

export default function Projects() {
  const projects = [
    {
      title: "Pneumonia Detection System",
      description: "AI-powered chest X-ray analysis deployed at City General Hospital, reducing diagnosis time by 60% and improving accuracy to 98.3%",
      category: "Radiology",
      icon: "🫁",
      stats: { scans: "50,000+", accuracy: "98.3%", time: "2s" },
      gradient: "from-blue-500/20 to-cyan-500/20",
    },
    {
      title: "Multi-Pathology Screening",
      description: "Comprehensive AI system detecting 8 critical chest pathologies at Memorial Medical Center with 95%+ accuracy across all conditions",
      category: "Diagnostic Imaging",
      icon: "🔬",
      stats: { scans: "120,000+", accuracy: "95.2%", time: "1.8s" },
      gradient: "from-purple-500/20 to-pink-500/20",
    },
    {
      title: "Emergency Room Triage AI",
      description: "Real-time critical finding detection system at St. Mary's ER, prioritizing urgent cases and reducing wait times by 40%",
      category: "Emergency Care",
      icon: "🚨",
      stats: { scans: "35,000+", accuracy: "97.9%", time: "1.5s" },
      gradient: "from-red-500/20 to-orange-500/20",
    },
  ];

  return (
    <section id="projects" className="py-24 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <p className="text-sm text-accent font-semibold tracking-wider mb-4">SUCCESS STORIES</p>
          <h2 className="font-display text-5xl md:text-6xl font-bold mb-6">
            Real-World <span className="text-accent">Impact</span>
          </h2>
          <p className="text-xl text-muted max-w-3xl mx-auto">
            Trusted by leading healthcare institutions to deliver faster, more accurate diagnoses
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {projects.map((project, index) => (
            <div
              key={index}
              className="group bg-white border border-border rounded-3xl overflow-hidden card-hover"
            >
              {/* Image Section with Icon */}
              <div className={`relative h-64 bg-gradient-to-br ${project.gradient} flex items-center justify-center overflow-hidden`}>
                {/* Background Pattern */}
                <div className="absolute inset-0 opacity-10">
                  <div className="absolute inset-0" style={{
                    backgroundImage: 'radial-gradient(circle, currentColor 1px, transparent 1px)',
                    backgroundSize: '20px 20px'
                  }}></div>
                </div>
                
                {/* Large Icon */}
                <div className="relative z-10 text-9xl group-hover:scale-110 transition-transform duration-500">
                  {project.icon}
                </div>

                {/* Floating Stats */}
                <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm rounded-xl px-3 py-2 shadow-lg">
                  <div className="text-xs text-muted">Scans Processed</div>
                  <div className="text-lg font-bold text-accent">{project.stats.scans}</div>
                </div>

                <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur-sm rounded-xl px-3 py-2 shadow-lg">
                  <div className="text-xs text-muted">Avg. Time</div>
                  <div className="text-lg font-bold text-green-600">{project.stats.time}</div>
                </div>
              </div>

              {/* Content Section */}
              <div className="p-8">
                <div className="flex items-center justify-between mb-4">
                  <span className="inline-block bg-accent/10 text-accent px-3 py-1 rounded-full text-xs font-semibold">
                    {project.category}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-2xl font-bold text-accent">{project.stats.accuracy}</span>
                    <span className="text-xs text-muted">Accuracy</span>
                  </div>
                </div>

                <h3 className="text-2xl font-bold mb-3 group-hover:text-accent transition-colors">
                  {project.title}
                </h3>
                
                <p className="text-muted leading-relaxed mb-6">
                  {project.description}
                </p>

                {/* Key Metrics */}
                <div className="grid grid-cols-3 gap-3 pt-6 border-t border-border">
                  <div className="text-center">
                    <div className="text-2xl mb-1">⚡</div>
                    <div className="text-xs text-muted">Fast</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl mb-1">🎯</div>
                    <div className="text-xs text-muted">Accurate</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl mb-1">✓</div>
                    <div className="text-xs text-muted">Reliable</div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Bottom CTA */}
        <div className="mt-16 text-center">
          <a 
            href="/contact" 
            className="inline-flex items-center gap-2 bg-accent text-white px-8 py-4 rounded-full font-semibold hover:bg-accent-dark transition-all hover:scale-105 shadow-lg shadow-accent/30"
          >
            Start Your Success Story
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </a>
        </div>
      </div>
    </section>
  );
}
