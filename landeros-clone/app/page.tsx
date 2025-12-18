import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Process from "@/components/Process";
import Projects from "@/components/Projects";
import Services from "@/components/Services";
import Testimonials from "@/components/Testimonials";
import Integrations from "@/components/Integrations";
import Footer from "@/components/Footer";
import TickerTape from "@/components/TickerTape";
import SimpleScanner from "@/components/SimpleScanner";
import ClinicalSupport from "@/components/ClinicalSupport";

export default function Home() {
  return (
    <main className="min-h-screen">
      <Navbar />
      <Hero />
      
      {/* Simple AI Demo */}
      <section className="py-24 px-6 bg-gradient-to-b from-white to-gray-50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <span className="inline-block bg-accent/10 text-accent px-4 py-2 rounded-full text-sm font-semibold mb-4">
              ⚡ LIVE DEMO
            </span>
            <h2 className="text-5xl md:text-6xl font-bold mb-6">
              Watch AI in <span className="text-accent">Action</span>
            </h2>
            <p className="text-xl text-muted max-w-3xl mx-auto">
              See how Neuron analyzes medical images in seconds with clinical-grade accuracy
            </p>
          </div>
          <SimpleScanner />
        </div>
      </section>

      {/* Clinical Decision Support */}
      <ClinicalSupport />

      <Process />
      <Projects />
      <Services />
      <TickerTape />
      <Testimonials />
      <Integrations />
      <Footer />
    </main>
  );
}
