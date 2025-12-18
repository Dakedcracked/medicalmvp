"use client";

export default function Strategy() {
  return (
    <section className="py-24 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="font-display text-5xl md:text-6xl font-bold mb-12">
            Strategy & Content Creation
          </h2>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <div className="bg-card border border-border rounded-2xl p-8 card-hover">
            <h3 className="text-3xl font-bold mb-4">AI Consulting & Strategies</h3>
            <p className="text-muted mb-8">
              We design custom AI strategies that drives growth.
            </p>
            
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="text-center">
                  <p className="text-sm text-muted mb-2">Before</p>
                  <div className="bg-background border border-border rounded-xl p-6">
                    <p className="text-2xl font-bold text-muted">Baseline</p>
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                <div className="text-center">
                  <p className="text-sm text-accent mb-2">After</p>
                  <div className="bg-accent/10 border border-accent rounded-xl p-6">
                    <p className="text-sm font-semibold text-accent mb-1">Growth +250%</p>
                    <p className="text-sm font-semibold text-accent mb-1">Efficiency +200%</p>
                    <p className="text-sm font-semibold text-accent">Cost -100%</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-card border border-border rounded-2xl p-8 card-hover">
            <h3 className="text-3xl font-bold mb-4">Social Media Content Creation</h3>
            <p className="text-muted mb-8">
              Leverage AI to create engaging, personalized content.
            </p>
            
            <div className="flex flex-col gap-4">
              <a
                href="#"
                onClick={(e) => {
                  e.preventDefault();
                  console.log("Book A Free Call Now clicked");
                }}
                className="bg-accent text-background px-8 py-4 rounded-full text-center font-semibold hover:bg-accent/90 transition-all hover:scale-105"
              >
                Book A Free Call Now
              </a>
              
              <a
                href="#contact"
                className="bg-transparent border border-accent text-accent px-8 py-4 rounded-full text-center font-semibold hover:bg-accent/10 transition-all"
              >
                Contact Sales Now
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
