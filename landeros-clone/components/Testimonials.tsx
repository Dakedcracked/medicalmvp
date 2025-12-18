"use client";

export default function Testimonials() {
  const testimonials = [
    {
      name: "Dr. Rajesh Kumar",
      company: "Apollo Hospitals",
      text: "Neuron's AI has significantly reduced our reporting turnaround time. The accuracy in detecting early-stage anomalies is truly impressive.",
    },
    {
      name: "Dr. Ananya Sharma",
      company: "Max Healthcare",
      text: "A revolutionary tool for radiologists. The interface is intuitive, and the 'second opinion' feature adds a layer of confidence to our diagnoses.",
    },
    {
      name: "Dr. Vikram Singh",
      company: "AIIMS Delhi",
      text: "We've integrated this into our resident training program. It's not just an analysis tool; it's an incredible learning aid for junior doctors.",
    },
    {
      name: "Priya Patel",
      company: "Star Imaging Centre",
      text: "The platform's ability to handle high volumes of X-rays without compromising speed or accuracy has transformed our workflow completely.",
    },
  ];

  return (
    <section id="reviews" className="py-24 px-6 bg-background-dark">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <p className="text-sm text-accent font-semibold tracking-wider mb-4">TESTIMONIALS</p>
          <h2 className="font-display text-5xl md:text-6xl font-bold mb-6">
            Trusted by Top <span className="text-accent">Doctors</span>
          </h2>
          <p className="text-lg text-muted">
            See what healthcare professionals across India are saying about Neuron
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {testimonials.map((testimonial, index) => (
            <div
              key={index}
              className="bg-white border border-border rounded-2xl p-6 card-hover shadow-sm"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 bg-gradient-to-br from-accent to-accent-dark rounded-full flex items-center justify-center shadow-md shadow-accent/20">
                  <span className="text-white font-bold text-lg">
                    {testimonial.name.charAt(0).toUpperCase()}
                  </span>
                </div>
                <div>
                  <h3 className="font-bold text-sm text-foreground">{testimonial.name}</h3>
                  <p className="text-xs text-accent font-medium">{testimonial.company}</p>
                </div>
              </div>
              <p className="text-sm text-muted leading-relaxed italic">"{testimonial.text}"</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
