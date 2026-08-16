import UploadMatcher from "./upload-matcher";

export default function Page() {
  return (
    <main className="flex flex-1 flex-col items-center gap-8 px-6 py-16">
      <div className="text-center">
        <h1 className="text-3xl font-semibold tracking-tight">
          Celebrity Face Match
        </h1>
        <p className="mt-2 text-sm text-neutral-500">
          Upload a selfie to find your closest celebrity look-alike.
        </p>
      </div>
      <UploadMatcher />
    </main>
  );
}
