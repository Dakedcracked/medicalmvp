"use client";

export default function CTA() {
  return (
    <section className="py-24 px-6">
      <div className="max-w-4xl mx-auto text-center">
        <div className="bg-gradient-to-br from-accent/20 to-accent/5 border border-accent/30 rounded-3xl p-12 md:p-16">
          <h2 className="font-display text-4xl md:text-5xl font-bold mb-6">
            " We Know your problems. We know your target audience and how you can grow rapidly with the help of automation „
          </h2>
          <p className="text-muted mb-8">Co-founder at landerOS</p>
          
          <div className="space-y-6">
            <div className="bg-card border border-border rounded-2xl p-6 text-left">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="font-bold text-lg mb-2">Product designer</h3>
                  <p className="text-sm text-muted">
                    looking for a product designer who really cares about the user experience 
                    and a team player who shapes our product
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-4 text-sm text-muted">
                <span>2+ years exp</span>
                <span>•</span>
                <span>Remote</span>
              </div>
            </div>

            <div className="bg-card border border-border rounded-2xl p-6 text-left">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="font-bold text-lg mb-2">Back end developer</h3>
                  <p className="text-sm text-muted">
                    We're looking for an experienced backend developer to build scalable systems
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-4 text-sm text-muted">
                <span>3+ years exp</span>
                <span>•</span>
                <span>Remote</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
