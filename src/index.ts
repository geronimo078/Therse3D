export { Engine, type EngineOptions, type EnginePreset, type EnginePlugin, type TickCallback } from './core/Engine';
export { vec3, quat, mat4, type Vec3, type Quat, type Mat4 } from './core/Math';
export { WorldStore, DirtyFlag, GeometryTypeId, LightKindId, MAX_ENTITIES_DEFAULT, MAX_MORPH_TARGETS } from './core/WorldStore';
export {
  SpatialLayout,
  type SpatialAlign,
  type SpatialPlacement,
  type AABB,
  type LayoutNodeInput,
} from './core/SpatialLayout';
export {
  SceneSync,
  type SceneSyncNode,
  type SceneSyncState,
  type ScenePatchState,
  type SceneSyncResult,
  type SceneSyncCamera,
  type CameraAnglePreset,
  type SceneSyncHooks,
} from './core/SceneSync';

export { GPUDeviceContext, type GPUDeviceInitOptions, type QualityMode } from './gpu/Device';
export { Renderer, type BloomConfig, type RendererOptions, type VolumetricConfig } from './gpu/Renderer';
export { BufferManager } from './gpu/BufferManager';

export { EntityHandle, type Vec3Like, type Vec4Like } from './entities/EntityHandle';
export {
  Box,
  Sphere,
  Cylinder,
  Plane,
  Line,
  Model,
  Group,
  Light,
  Cone,
  Torus,
  createBoxGeometry,
  createSphereGeometry,
  createCylinderGeometry,
  createPlaneGeometry,
  createLineGeometry,
  createConeGeometry,
  createTorusGeometry,
  type PrimitiveOptions,
  type PrimitiveType,
  type LightKind,
  type PrimitiveDescriptor,
  type GeometryMeshData,
  type MorphTargetData,
} from './entities/Primitives';
export {
  MATERIAL_PRESETS,
  resolveMaterial,
  parseColor,
  type MaterialDefinition,
  type MaterialPresetName,
} from './entities/Materials';
export { GltfLoader, type GltfLoadResult, type GltfParsedMesh } from './entities/GltfLoader';

export {
  Camera,
  OrbitControls,
  OrbitControlsController,
  type OrbitControlsOptions,
  type FitAllOptions,
} from './controls/OrbitControls';

export {
  Inspector,
  raycastStore,
  pickFromScreen,
  projectWorldPoint,
  type RaycastHit,
  type ProjectedPoint,
  type SuggestedFix,
  type EntityDescription,
  type SceneDescription,
  type ScreenRegion,
  type EntityRelations,
  type DescribeContext,
} from './ai/Inspector';
export { Snapshot, type SnapshotOptions, type SnapshotResult } from './ai/Snapshot';
export { type SnapshotLayer, type CapturePassResult } from './gpu/Renderer';
export {
  Schema,
  THERSE3D_SCENE_SCHEMA,
  THERSE3D_TOOL_DEFINITIONS,
  SCENE_SYNC_JSON_SCHEMA,
} from './ai/Schema';
export {
  McpBridge,
  type McpBridgeOptions,
  type McpToolName,
  type McpToolRequest,
  type CameraFocusArgs,
} from './ai/McpBridge';

