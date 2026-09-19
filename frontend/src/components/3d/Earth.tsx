import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export const Earth: React.FC = () => {
  const earthRef = useRef<THREE.Group>(null);
  
  useFrame((_, delta) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += delta * 0.05; // Slow idle rotation
    }
  });

  return (
    <group ref={earthRef}>
      {/* Core Earth Sphere (Dark) */}
      <mesh>
        <sphereGeometry args={[2, 64, 64]} />
        <meshStandardMaterial 
          color="#020813"
          roughness={0.8}
          metalness={0.2}
          emissive="#001122"
        />
      </mesh>
      
      {/* Wireframe Grid / Lat-Lon */}
      <mesh>
        <sphereGeometry args={[2.01, 32, 32]} />
        <meshBasicMaterial 
          color="#37E6B4" 
          wireframe 
          transparent 
          opacity={0.12} 
        />
      </mesh>

      {/* Atmosphere Glow */}
      <mesh>
        <sphereGeometry args={[2.12, 64, 64]} />
        <meshBasicMaterial 
          color="#37E6B4" 
          transparent 
          opacity={0.06}
          side={THREE.BackSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </group>
  );
};
