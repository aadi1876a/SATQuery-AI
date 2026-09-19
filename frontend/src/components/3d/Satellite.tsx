import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface SatelliteProps {
  radius?: number;
  /** Radians per second. One full orbit = 2π / speed seconds.
   *  Default 0.12 rad/s → ~52 s per orbit (slow, ambient "live monitoring" pace). */
  speed?: number;
}

export const Satellite: React.FC<SatelliteProps> = ({ radius = 2.5, speed = 0.12 }) => {
  const satelliteRef = useRef<THREE.Group>(null);

  useFrame(({ clock }) => {
    if (satelliteRef.current) {
      // Use absolute elapsed time so speed is consistent at any framerate.
      // No per-frame increment — entirely clock driven (delta-time safe).
      const t = clock.getElapsedTime() * speed;

      // Constant-speed circular orbit (no easing on the continuous loop —
      // easing only belongs on one-time transitions, not repeating loops).
      satelliteRef.current.position.x = Math.sin(t) * radius;
      satelliteRef.current.position.z = Math.cos(t) * radius;

      // Slight orbital inclination tied to the same orbit angle (not an
      // independent slower oscillation), so the tilt stays proportional
      // and motion feels uniform at any playback speed.
      satelliteRef.current.position.y = Math.sin(t) * (radius * 0.18);

      // Always face the center of the Earth
      satelliteRef.current.lookAt(0, 0, 0);
    }
  });

  return (
    <group ref={satelliteRef}>
      {/* Satellite Body */}
      <mesh>
        <boxGeometry args={[0.1, 0.1, 0.15]} />
        <meshStandardMaterial color="#ffffff" metalness={0.8} roughness={0.2} />
      </mesh>

      {/* Solar Panel 1 */}
      <mesh position={[0.15, 0, 0]}>
        <boxGeometry args={[0.2, 0.02, 0.1]} />
        <meshStandardMaterial color="#0088cc" metalness={0.5} roughness={0.5} />
      </mesh>

      {/* Solar Panel 2 */}
      <mesh position={[-0.15, 0, 0]}>
        <boxGeometry args={[0.2, 0.02, 0.1]} />
        <meshStandardMaterial color="#0088cc" metalness={0.5} roughness={0.5} />
      </mesh>

      {/* Subtle indicator light */}
      <mesh position={[0, 0.06, 0]}>
        <sphereGeometry args={[0.02, 8, 8]} />
        <meshBasicMaterial color="#00d4ff" />
      </mesh>
    </group>
  );
};
