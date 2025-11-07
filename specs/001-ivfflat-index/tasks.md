# Tasks: IVFFLAT Index Implementation

**Input**: Design documents from `/specs/001-ivfflat-index/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/api-contract.md, quickstart.md

**Tests**: Tests are OPTIONAL for this feature - include only if explicitly requested later.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- Single project structure: `src/my_vector_db/`, `tests/` at repository root
- All paths are relative to `/Users/adamshedivy/Documents/projects/my-vector-db/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Verify Python 3.11+ and uv package manager are installed
- [x] T002 Run `uv sync` to install all project dependencies including numpy and scikit-learn
- [x] T003 [P] Review existing VectorIndex interface in src/my_vector_db/indexes/base.py
- [x] T004 [P] Review existing FLAT index implementation in src/my_vector_db/indexes/flat.py for patterns

**Checkpoint**: Environment ready, existing code understood

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core changes that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T005 Add IndexType.IVF = "ivf" to IndexType enum in src/my_vector_db/domain/models.py
- [x] T006 Run existing unit tests to verify backward compatibility with new enum value: `uv run pytest tests/unit/test_models.py -v`

**Checkpoint**: Foundation ready - IVF is now a valid index type throughout the system. User story implementation can now begin.

---

## Phase 3: User Story 1 - Index Creation with Clustering (Priority: P1) 🎯 MVP

**Goal**: Implement the foundation of IVF functionality - cluster-based vector organization using K-means clustering, with support for lazy building on first search OR explicit build_index call.

**Independent Test**: Create a library with index_type="ivf", add vectors, and verify clustering occurs either (a) after explicit build_index call, or (b) automatically on first search. Delivers a functional clustered index structure.

**User Story 1 Acceptance Criteria**:
1. Library created with IVF index using existing create_library method
2. Vectors assigned to clusters after build_index call
3. Clustering triggers automatically on first search if not built
4. Rebuild recomputes clustering and reassigns vectors
5. Dimension validation enforced

### Implementation for User Story 1

- [x] T007 [US1] Create src/my_vector_db/indexes/ivf.py with IVFIndex class skeleton implementing VectorIndex interface
- [x] T008 [US1] Implement IVFIndex.__init__() with config validation (nlist, nprobe, metric) and instance variables (_vectors, _clusters, _centroids, _kmeans, _is_built)
- [x] T009 [US1] Implement IVFIndex.add() method with dimension validation and conditional cluster assignment (assign to nearest cluster if built, otherwise queue for clustering)
- [x] T010 [US1] Implement IVFIndex.bulk_add() method that delegates to add() for each vector
- [x] T011 [US1] Implement IVFIndex.delete() method to remove vectors from _vectors and _clusters
- [x] T012 [US1] Implement IVFIndex.update() method using delete() then add() pattern
- [x] T013 [US1] Implement IVFIndex.clear() method to reset all index state
- [x] T014 [US1] Implement IVFIndex._compute_default_nlist() helper method using sqrt(n) formula per research.md
- [ ] T015 [US1] Implement IVFIndex._build_clusters() method with K-means clustering using scikit-learn (n_clusters=nlist, random_state=42, n_init=10, max_iter=300)
- [ ] T016 [US1] Implement cluster assignment logic in _build_clusters() to populate _clusters dict with (vector_id, vector) tuples per research.md
- [ ] T017 [US1] Implement IVFIndex._find_nearest_cluster() method using centroid distance computation with configured metric (cosine, euclidean, dot_product)
- [x] T018 [US1] Implement IVFIndex.build() public method that calls _build_clusters() for explicit clustering trigger
- [ ] T019 [US1] Update LibraryService._create_index() factory in src/my_vector_db/services/library_service.py to handle IndexType.IVF and return IVFIndex instance

**Checkpoint**: At this point, User Story 1 should be fully functional:
- IVF libraries can be created via SDK with index_type="ivf"
- Vectors can be added and assigned to clusters
- Explicit build_index() triggers clustering
- Basic CRUD operations (add, update, delete, clear) work correctly

---

## Phase 4: User Story 2 - Fast Approximate Search (Priority: P2)

**Goal**: Deliver the primary value proposition of IVF - faster search through cluster-based pruning. The system identifies the most relevant clusters (using nprobe) and only searches within those clusters instead of comparing against all vectors.

**Independent Test**: Perform searches on an IVF index and compare execution time and recall against FLAT index on the same data. Delivers measurable performance improvement on 10,000+ vector datasets.

**User Story 2 Acceptance Criteria**:
1. Most relevant clusters identified based on nprobe configuration
2. Search returns results from nearest clusters only
3. System handles cases where clusters contain fewer vectors than k
4. Exactly nprobe clusters searched per query

### Implementation for User Story 2

- [ ] T020 [US2] Implement IVFIndex.search() method with lazy build trigger (call _build_clusters() if not _is_built)
- [ ] T021 [US2] Implement IVFIndex._get_nprobe_nearest_clusters() method to identify nprobe nearest clusters by centroid similarity
- [ ] T022 [US2] Add empty cluster skipping logic in _get_nprobe_nearest_clusters() per research.md
- [ ] T023 [US2] Implement candidate collection logic in search() to gather vectors from selected clusters
- [ ] T024 [US2] Implement similarity scoring in search() using configured metric (cosine, euclidean, dot_product) from VectorIndex base class
- [ ] T025 [US2] Implement result sorting and top-k selection in search() method
- [ ] T026 [US2] Add nprobe clamping logic (min(nprobe, nlist)) to handle nprobe > nlist edge case per spec.md

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently:
- Searches return approximate results based on cluster pruning
- Lazy building works (clustering on first search if not built)
- Search performance faster than FLAT on datasets with 10,000+ vectors
- nprobe parameter controls speed/accuracy trade-off

---

## Phase 5: User Story 3 - Configurable Search Quality (Priority: P3)

**Goal**: Provide fine-tuning capability for advanced users to control the speed/accuracy trade-off by configuring nprobe at library creation or updating it via update_library. More clusters searched (higher nprobe) means higher accuracy but slower speed.

**Independent Test**: Create libraries with different nprobe settings, perform searches, and measure recall percentage and query time. Test updating nprobe via update_library and observe changed search behavior.

**User Story 3 Acceptance Criteria**:
1. nprobe=1 searches only 1 cluster
2. nprobe=10 searches 10 nearest clusters and combines results
3. nprobe > nlist searches all clusters (exhaustive)
4. Updating nprobe via update_library changes subsequent search behavior

### Implementation for User Story 3

- [ ] T027 [US3] Verify nprobe configuration is read from self.config in search() and _get_nprobe_nearest_clusters() methods
- [ ] T028 [US3] Add nprobe validation in IVFIndex.__init__() to ensure positive integer (raise ValueError if invalid)
- [ ] T029 [US3] Verify Library.index_config accepts nprobe parameter and persists it correctly (test with existing Pydantic validation)
- [ ] T030 [US3] Test update_library() endpoint to verify nprobe can be updated and affects subsequent searches (integration test via SDK)

**Checkpoint**: At this point, User Stories 1, 2, AND 3 should all work independently:
- Users can configure nprobe at library creation time
- Users can update nprobe via update_library without rebuild
- Different nprobe values demonstrate measurable speed/accuracy trade-offs
- All edge cases handled (nprobe=0, nprobe > nlist)

---

## Phase 6: User Story 4 - Index Maintenance Operations (Priority: P4)

**Goal**: Enable practical use of IVF in dynamic environments where data changes over time. Support vector updates (reassigning to appropriate cluster if needed) and deletions without requiring full index rebuild.

**Independent Test**: Perform add/update/delete operations on an IVF index and verify vectors are correctly maintained in clusters without requiring rebuild.

**User Story 4 Acceptance Criteria**:
1. Updated vectors reassigned to appropriate cluster
2. Deleted vectors removed from clusters and no longer in search results
3. New vectors added after clustering assigned to nearest cluster without full reclustering

### Implementation for User Story 4

- [ ] T031 [US4] Verify IVFIndex.add() correctly handles post-clustering vector additions by assigning to nearest cluster (already implemented in T009, add validation test)
- [ ] T032 [US4] Verify IVFIndex.update() correctly reassigns updated vectors to appropriate cluster (already implemented in T012, add validation test)
- [ ] T033 [US4] Verify IVFIndex.delete() correctly removes vectors from clusters and handles empty cluster edge case (already implemented in T011, add validation test)
- [ ] T034 [US4] Test bulk operations after clustering to verify incremental behavior without rebuild (bulk_add after build)

**Checkpoint**: All user stories should now be independently functional:
- Full CRUD operations work correctly with cluster maintenance
- Empty clusters handled gracefully
- No rebuild required for incremental operations
- Production-ready mutable index

---

## Phase 7: Testing & Validation (Cross-Cutting)

**Purpose**: Comprehensive validation across all user stories

**Note**: These tests are OPTIONAL - only create if explicitly requested or if following TDD approach.

- [ ] T035 [P] Create tests/unit/test_ivf_index.py with unit tests for clustering, search, and maintenance operations
- [ ] T036 [P] Create tests/integration/test_ivf_search.py with end-to-end workflow tests
- [ ] T037 [P] Create tests/contract/test_ivf_contract.py to verify VectorIndex interface compliance
- [ ] T038 Run full test suite: `uv run pytest -v`
- [ ] T039 Check test coverage (must be >80% per constitution): `uv run pytest --cov=my_vector_db --cov-report=term-missing`
- [ ] T040 Run linting: `uv run ruff check`
- [ ] T041 Run formatting check: `uv run ruff format --check`

**Checkpoint**: All constitutional requirements met (>80% coverage, Pure Python, Type Safety, Thread Safety)

---

## Phase 8: Documentation & Polish

**Purpose**: Finalize documentation and examples

- [ ] T042 [P] Update README.md with IVF index usage example showing nlist/nprobe configuration
- [ ] T043 [P] Add performance comparison section to README.md (IVF vs FLAT benchmarks)
- [ ] T044 [P] Update OpenAPI documentation with IVF examples in FastAPI route docstrings
- [ ] T045 [P] Verify /docs endpoint shows IVF configuration parameters correctly
- [ ] T046 Add performance tuning recommendations to documentation (nlist and nprobe guidelines from quickstart.md)
- [ ] T047 Create migration guide for users wanting to convert FLAT libraries to IVF
- [ ] T048 Review all generated code for code quality, comments, and maintainability

**Checkpoint**: Feature complete, documented, and ready for production use

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User Story 1 (P1) MUST complete before User Story 2 (P2) - US2 depends on clustering from US1
  - User Story 2 (P2) CAN proceed in parallel with User Story 3 (P3) - independent
  - User Story 2 (P2) MUST complete before User Story 4 (P4) - US4 tests maintenance during search
  - User Story 3 (P3) CAN proceed in parallel with User Story 4 (P4) - independent
- **Testing (Phase 7)**: Depends on all user stories being complete (optional phase)
- **Documentation (Phase 8)**: Depends on all implementation being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories - FOUNDATION
- **User Story 2 (P2)**: Depends on User Story 1 completion (needs clustering) - CORE VALUE
- **User Story 3 (P3)**: Depends on User Story 2 completion (needs search) - TUNING
- **User Story 4 (P4)**: Depends on User Story 1 completion (needs clustering) - MAINTENANCE

### Critical Path

**Minimum viable implementation path**:
1. Setup (Phase 1) → Foundational (Phase 2) → User Story 1 (Phase 3) → User Story 2 (Phase 4)

This delivers:
- IVF index creation with clustering ✅
- Fast approximate search ✅
- Performance improvement over FLAT ✅

### Within Each User Story

- Tasks within a story must follow the order listed (sequential dependencies)
- Tasks marked [P] in different stories can run in parallel
- Complete all tasks in a story before moving to next priority

### Parallel Opportunities

Within phases:
- T003 and T004 (Setup) can run in parallel
- T035, T036, T037 (Testing) can run in parallel
- T042, T043, T044, T045 (Documentation) can run in parallel

Across phases (if team capacity allows):
- User Story 3 (P3) and User Story 4 (P4) can be worked on in parallel after US1 and US2 complete

---

## Parallel Example: Setup Phase

```bash
# Launch setup tasks together:
Task T003: "Review existing VectorIndex interface in src/my_vector_db/indexes/base.py"
Task T004: "Review existing FLAT index implementation in src/my_vector_db/indexes/flat.py"
```

## Parallel Example: Documentation Phase

```bash
# Launch documentation tasks together:
Task T042: "Update README.md with IVF index usage example"
Task T043: "Add performance comparison section to README.md"
Task T044: "Update OpenAPI documentation with IVF examples"
Task T045: "Verify /docs endpoint shows IVF configuration parameters"
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2 Only)

**Minimum viable product delivers core IVF value:**

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - adds IVF to IndexType enum)
3. Complete Phase 3: User Story 1 (clustering foundation)
4. Complete Phase 4: User Story 2 (fast approximate search)
5. **STOP and VALIDATE**: Test US1 and US2 independently
6. Deploy/demo if ready

**Result**: Users can create IVF libraries and perform fast approximate searches on large datasets.

### Incremental Delivery

1. **Foundation** (Phases 1-2) → IVF is valid index type
2. **MVP** (Phases 3-4) → Clustering + Search working → Deploy/Demo ✅
3. **Tuning** (Phase 5) → Add nprobe configuration → Deploy/Demo ✅
4. **Maintenance** (Phase 6) → Add CRUD operations → Deploy/Demo ✅
5. **Polish** (Phases 7-8) → Tests + Documentation → Production Ready ✅

Each increment adds value without breaking previous functionality.

### Parallel Team Strategy

With multiple developers after Foundational phase completes:

1. **Team completes Setup + Foundational together** (Phases 1-2)
2. **Once Foundational is done**:
   - Developer A: User Story 1 (Phase 3) → MUST complete first
   - Developer A continues: User Story 2 (Phase 4) → Depends on US1
3. **After US1 and US2 complete**:
   - Developer B: User Story 3 (Phase 5) → Can start
   - Developer C: User Story 4 (Phase 6) → Can start in parallel
4. **After all stories complete**:
   - Multiple developers: Testing (Phase 7) in parallel
   - Multiple developers: Documentation (Phase 8) in parallel

---

## Task Summary

**Total Tasks**: 48
**Tasks by User Story**:
- Setup: 4 tasks
- Foundational: 2 tasks
- User Story 1 (P1): 13 tasks - Clustering foundation
- User Story 2 (P2): 7 tasks - Search implementation
- User Story 3 (P3): 4 tasks - Configuration tuning
- User Story 4 (P4): 4 tasks - Maintenance operations
- Testing: 7 tasks (optional)
- Documentation: 7 tasks

**Parallel Opportunities Identified**: 12 tasks marked [P]
- 2 in Setup phase
- 3 in Testing phase
- 5 in Documentation phase
- 2 cross-story opportunities (US3 + US4 after US1/US2)

**Suggested MVP Scope**: Phases 1-4 (Setup + Foundational + US1 + US2) = 26 tasks
**MVP Delivers**: IVF index creation, clustering, and fast approximate search

**Independent Test Criteria**:
- ✅ US1: Create library, add vectors, build index, verify clustering
- ✅ US2: Perform search, measure time vs FLAT, verify approximate results
- ✅ US3: Configure different nprobe values, measure recall differences
- ✅ US4: Add/update/delete after clustering, verify no rebuild required

**Format Validation**: ✅ All tasks follow checklist format with Task ID, optional [P] marker, [Story] label (where applicable), and exact file paths

---

## Notes

- All tasks follow the required format: `- [ ] [ID] [P?] [Story?] Description with file path`
- [P] tasks target different files with no dependencies
- [Story] labels (US1, US2, US3, US4) map tasks to specific user stories
- Each user story is independently completable and testable
- Tests are OPTIONAL - only included in Phase 7 (not required by spec)
- Stop at any checkpoint to validate story independently
- Follow quickstart.md for detailed implementation guidance
- Commit after each task or logical group
- Run `uv run pytest -v` at each checkpoint to validate changes

---

**Ready for implementation!** Follow phases sequentially, test at each checkpoint, and deliver incrementally per constitutional Principle VII.
