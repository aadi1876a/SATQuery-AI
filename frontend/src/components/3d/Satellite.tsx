import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface SatelliteProps {
  radius?: number;
  speed?: number;
}

export const Satellite: React.FC<SatelliteProps> = ({ radius = 2.5, speed = 0.5 }) => {
  const satelliteRef = useRef<THREE.Group>(null);
  
  useFrame(({ clock }) => {
    if (satelliteRef.current) {
      const t = clock.getElapsedTime() * speed;
      // Orbit path
      satelliteRef.current.position.x = Math.sin(t) * radius;
      satelliteRef.current.position.z = Math.cos(t) * radius;
      // Tilt the orbit slightly
      satelliteRef.current.position.y = Math.sin(t * 0.5) * (radius * 0.2);
      
      // Face direction of movement
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
