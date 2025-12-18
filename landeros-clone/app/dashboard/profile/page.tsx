"use client";

import { useEffect, useState } from "react";
import { AuthService, User } from "@/lib/auth";
import { useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export default function ProfilePage() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    specialty: "",
    hospital: "",
    phone: "",
    location: "",
  });
  const router = useRouter();

  useEffect(() => {
    const checkAuth = async () => {
      const authenticated = AuthService.isAuthenticated();
      if (!authenticated) {
        router.push("/");
        return;
      }

      const currentUser = AuthService.getUser();
      setUser(currentUser);
      setFormData({
        name: currentUser?.name || "",
        email: currentUser?.email || "",
        specialty: "Radiology",
        hospital: "General Hospital",
        phone: "+1 (555) 123-4567",
        location: "New York, USA",
      });
      setLoading(false);
    };

    checkAuth();
  }, [router]);

  const handleSave = () => {
    // Update user info
    if (user) {
      const updatedUser = { ...user, name: formData.name };
      AuthService.setUser(updatedUser);
      setUser(updatedUser);
    }
    setEditing(false);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-accent border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-muted">Loading profile...</p>
        </div>
      </div>
    );
  }

  return (
    <main className="min-h-screen relative">
      <div className="relative z-10">
        <Navbar />
        
        <div className="pt-32 pb-20 px-6">
          <div className="max-w-5xl mx-auto">
            {/* Header */}
            <div className="mb-8">
              <a href="/dashboard" className="inline-flex items-center gap-2 text-muted hover:text-accent transition-colors mb-4">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Back to Dashboard
              </a>
              <h1 className="text-5xl font-bold mb-2">Profile Settings</h1>
              <p className="text-muted text-lg">Manage your account information and preferences</p>
            </div>

            <div className="grid lg:grid-cols-3 gap-8">
              {/* Sidebar */}
              <div className="lg:col-span-1">
                <div className="bg-white rounded-2xl p-8 border border-border text-center">
                  <div className="w-32 h-32 bg-gradient-to-br from-accent to-accent-dark rounded-full flex items-center justify-center mx-auto mb-4">
                    <span className="text-6xl text-white font-bold">
                      {user?.name.charAt(0).toUpperCase()}
                    </span>
                  </div>
                  <h2 className="text-2xl font-bold mb-2">{user?.name}</h2>
                  <p className="text-muted mb-4">{user?.email}</p>
                  
                  <div className="space-y-2 mb-6">
                    <div className="bg-background-dark p-3 rounded-xl">
                      <p className="text-sm text-muted">Member Since</p>
                      <p className="font-semibold">Nov 2024</p>
                    </div>
                    <div className="bg-background-dark p-3 rounded-xl">
                      <p className="text-sm text-muted">Account Type</p>
                      <p className="font-semibold">Professional</p>
                    </div>
                    <div className="bg-background-dark p-3 rounded-xl">
                      <p className="text-sm text-muted">Status</p>
                      <p className="font-semibold text-green-600">Active</p>
                    </div>
                  </div>

                  <button className="w-full bg-accent text-white py-3 rounded-full font-semibold hover:bg-accent-dark transition-all">
                    Upload Photo
                  </button>
                </div>
              </div>

              {/* Main Content */}
              <div className="lg:col-span-2 space-y-6">
                {/* Personal Information */}
                <div className="bg-white rounded-2xl p-8 border border-border">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-2xl font-bold">Personal Information</h3>
                    {!editing ? (
                      <button
                        onClick={() => setEditing(true)}
                        className="text-accent hover:text-accent-dark transition-colors font-semibold"
                      >
                        Edit
                      </button>
                    ) : (
                      <div className="flex gap-2">
                        <button
                          onClick={() => setEditing(false)}
                          className="px-4 py-2 border border-border rounded-lg hover:bg-background-dark transition-colors"
                        >
                          Cancel
                        </button>
                        <button
                          onClick={handleSave}
                          className="px-4 py-2 bg-accent text-white rounded-lg hover:bg-accent-dark transition-colors"
                        >
                          Save
                        </button>
                      </div>
                    )}
                  </div>

                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-semibold mb-2">Full Name</label>
                      <input
                        type="text"
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                        disabled={!editing}
                        className="w-full px-4 py-3 border border-border rounded-xl focus:outline-none focus:border-accent transition-colors disabled:bg-background-dark disabled:cursor-not-allowed"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-semibold mb-2">Email Address</label>
                      <input
                        type="email"
                        value={formData.email}
                        disabled
                        className="w-full px-4 py-3 border border-border rounded-xl bg-background-dark cursor-not-allowed"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-semibold mb-2">Specialty</label>
                      <input
                        type="text"
                        value={formData.specialty}
                        onChange={(e) => setFormData({ ...formData, specialty: e.target.value })}
                        disabled={!editing}
                        className="w-full px-4 py-3 border border-border rounded-xl focus:outline-none focus:border-accent transition-colors disabled:bg-background-dark disabled:cursor-not-allowed"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-semibold mb-2">Hospital/Clinic</label>
                      <input
                        type="text"
                        value={formData.hospital}
                        onChange={(e) => setFormData({ ...formData, hospital: e.target.value })}
                        disabled={!editing}
                        className="w-full px-4 py-3 border border-border rounded-xl focus:outline-none focus:border-accent transition-colors disabled:bg-background-dark disabled:cursor-not-allowed"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-semibold mb-2">Phone Number</label>
                      <input
                        type="tel"
                        value={formData.phone}
                        onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                        disabled={!editing}
                        className="w-full px-4 py-3 border border-border rounded-xl focus:outline-none focus:border-accent transition-colors disabled:bg-background-dark disabled:cursor-not-allowed"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-semibold mb-2">Location</label>
                      <input
                        type="text"
                        value={formData.location}
                        onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                        disabled={!editing}
                        className="w-full px-4 py-3 border border-border rounded-xl focus:outline-none focus:border-accent transition-colors disabled:bg-background-dark disabled:cursor-not-allowed"
                      />
                    </div>
                  </div>
                </div>

                {/* Account Security */}
                <div className="bg-white rounded-2xl p-8 border border-border">
                  <h3 className="text-2xl font-bold mb-6">Account Security</h3>
                  
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-background-dark rounded-xl">
                      <div>
                        <p className="font-semibold mb-1">Password</p>
                        <p className="text-sm text-muted">Last changed 30 days ago</p>
                      </div>
                      <button className="text-accent hover:text-accent-dark font-semibold">
                        Change
                      </button>
                    </div>

                    <div className="flex items-center justify-between p-4 bg-background-dark rounded-xl">
                      <div>
                        <p className="font-semibold mb-1">Two-Factor Authentication</p>
                        <p className="text-sm text-muted">Add an extra layer of security</p>
                      </div>
                      <button className="text-accent hover:text-accent-dark font-semibold">
                        Enable
                      </button>
                    </div>

                    <div className="flex items-center justify-between p-4 bg-background-dark rounded-xl">
                      <div>
                        <p className="font-semibold mb-1">Active Sessions</p>
                        <p className="text-sm text-muted">Manage your active sessions</p>
                      </div>
                      <button className="text-accent hover:text-accent-dark font-semibold">
                        View
                      </button>
                    </div>
                  </div>
                </div>

                {/* Preferences */}
                <div className="bg-white rounded-2xl p-8 border border-border">
                  <h3 className="text-2xl font-bold mb-6">Preferences</h3>
                  
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-background-dark rounded-xl">
                      <div>
                        <p className="font-semibold mb-1">Email Notifications</p>
                        <p className="text-sm text-muted">Receive analysis updates via email</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" className="sr-only peer" defaultChecked />
                        <div className="w-11 h-6 bg-gray-300 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-accent"></div>
                      </label>
                    </div>

                    <div className="flex items-center justify-between p-4 bg-background-dark rounded-xl">
                      <div>
                        <p className="font-semibold mb-1">Critical Alerts</p>
                        <p className="text-sm text-muted">Instant notifications for critical findings</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" className="sr-only peer" defaultChecked />
                        <div className="w-11 h-6 bg-gray-300 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-accent"></div>
                      </label>
                    </div>

                    <div className="flex items-center justify-between p-4 bg-background-dark rounded-xl">
                      <div>
                        <p className="font-semibold mb-1">Weekly Reports</p>
                        <p className="text-sm text-muted">Summary of your analysis activity</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" className="sr-only peer" />
                        <div className="w-11 h-6 bg-gray-300 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-accent"></div>
                      </label>
                    </div>
                  </div>
                </div>

                {/* Danger Zone */}
                <div className="bg-white rounded-2xl p-8 border border-red-200">
                  <h3 className="text-2xl font-bold mb-6 text-red-600">Danger Zone</h3>
                  
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-red-50 rounded-xl border border-red-200">
                      <div>
                        <p className="font-semibold mb-1">Delete Account</p>
                        <p className="text-sm text-muted">Permanently delete your account and all data</p>
                      </div>
                      <button className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-semibold">
                        Delete
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <Footer />
      </div>
    </main>
  );
}
