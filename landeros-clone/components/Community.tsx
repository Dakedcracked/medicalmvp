"use client";

import { useState } from "react";

export default function Contact() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    subject: "",
    message: "",
  });
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/contact", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error("Failed to send message");
      }

      setSuccess(true);
      setFormData({ name: "", email: "", subject: "", message: "" });
      setTimeout(() => setSuccess(false), 5000);
    } catch (err: any) {
      setError(err.message || "Failed to send message");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="contact" className="py-24 px-6">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="font-display text-5xl md:text-6xl font-bold mb-6">
            Get in <span className="text-accent">Touch</span>
          </h2>
          <p className="text-lg text-muted max-w-2xl mx-auto">
            Have questions about Neuron? We're here to help. Send us a message and we'll respond as soon as possible.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Contact Form */}
          <div className="bg-white rounded-2xl p-8 border border-border shadow-sm">
            <h3 className="text-2xl font-bold mb-6">Send us a message</h3>
            
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-semibold mb-2">Full Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  required
                  className="w-full px-4 py-3 border border-border rounded-xl focus:outline-none focus:border-accent transition-colors"
                  placeholder="Dr. Aditiya Prasad"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold mb-2">Email Address</label>
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  required
                  className="w-full px-4 py-3 border border-border rounded-xl focus:outline-none focus:border-accent transition-colors"
                  placeholder="contact@example.com"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold mb-2">Subject</label>
                <input
                  type="text"
                  value={formData.subject}
                  onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                  required
                  className="w-full px-4 py-3 border border-border rounded-xl focus:outline-none focus:border-accent transition-colors"
                  placeholder="Partnership Inquiry"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold mb-2">Message</label>
                <textarea
                  value={formData.message}
                  onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                  required
                  rows={5}
                  className="w-full px-4 py-3 border border-border rounded-xl focus:outline-none focus:border-accent transition-colors resize-none"
                  placeholder="Tell us about your needs..."
                />
              </div>

              {success && (
                <div className="bg-green-500/10 border border-green-500/50 rounded-xl p-4 text-green-600">
                  ✓ Message sent successfully! We'll get back to you soon.
                </div>
              )}

              {error && (
                <div className="bg-red-500/10 border border-red-500/50 rounded-xl p-4 text-red-600">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-accent text-white py-3 rounded-full font-semibold hover:bg-accent-dark transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-accent/20"
              >
                {loading ? "Sending..." : "Send Message"}
              </button>
            </form>
          </div>

          {/* Contact Info */}
          <div className="space-y-6">
            <div className="bg-white rounded-2xl p-8 border border-border shadow-sm hover:shadow-md transition-all">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 bg-accent/10 rounded-xl flex items-center justify-center flex-shrink-0 text-accent">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                </div>
                <div className="overflow-hidden">
                  <h4 className="font-bold mb-2 text-lg">Email Us</h4>
                  <a href="mailto:conversationwithvarun@gmail.com" className="block text-muted hover:text-accent transition-colors truncate">conversationwithvarun@gmail.com</a>
                  <a href="mailto:aditiyaprasad2023acc@gmail.com" className="block text-muted hover:text-accent transition-colors truncate">aditiyaprasad2023acc@gmail.com</a>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-8 border border-border shadow-sm hover:shadow-md transition-all">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 bg-accent/10 rounded-xl flex items-center justify-center flex-shrink-0 text-accent">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                  </svg>
                </div>
                <div>
                  <h4 className="font-bold mb-2 text-lg">Call Us</h4>
                  <p className="text-muted hover:text-accent transition-colors cursor-pointer">+91 8923820910</p>
                  <p className="text-muted hover:text-accent transition-colors cursor-pointer">+91 9674989132</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-8 border border-border shadow-sm hover:shadow-md transition-all">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 bg-accent/10 rounded-xl flex items-center justify-center flex-shrink-0 text-accent">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <div>
                  <h4 className="font-bold mb-2 text-lg">Visit Us</h4>
                  <p className="text-muted">South Asia University</p>
                  <p className="text-muted">11th Floor, A1 Building</p>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-accent to-accent-dark text-white rounded-2xl p-8 shadow-lg shadow-accent/20">
              <h4 className="font-bold mb-2 text-lg">Enterprise Solutions</h4>
              <p className="text-white/90 mb-6">
                Looking for custom AI solutions for your hospital or healthcare network?
              </p>
              <a href="mailto:conversationwithvarun@gmail.com" className="inline-flex items-center justify-center w-full bg-white text-accent px-6 py-3 rounded-full font-bold hover:bg-gray-50 transition-all transform hover:-translate-y-1">
                Contact Sales Team
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
